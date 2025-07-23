#!/usr/bin/env python3
"""
CMI Behavior Detection Training Script
Converts notebook to standalone training application
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from math import ceil
from scipy import stats
from scipy.fft import fft, fftfreq
from collections import Counter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise, Add, GlobalMaxPooling1D,
    MultiHeadAttention, LayerNormalization
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
import polars as pl
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuration
TRAIN = True
RAW_DIR = Path("./input")  # Update this to your data directory
OUTPUT_DIR = Path("./best_model")
BATCH_SIZE = 64
PAD_PERCENTILE = 90
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 120
PATIENCE = 40
N_SPLITS = 6

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Enable mixed precision training for GPU
if TRAIN:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)

# Tensor manipulation functions
def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

# Residual CNN Block with SE
def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

def transformer_block(x, head_size=64, num_heads=4, ff_dim=256, dropout=0.1):
    """Add a transformer block for sequence modeling"""
    # Multi-head attention
    attn_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=head_size,
        dropout=dropout
    )(x, x)
    attn_output = Dropout(dropout)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    # Feed forward
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dropout(dropout)(ffn_output)
    ffn_output = Dense(x.shape[-1])(ffn_output)
    ffn_output = Dropout(dropout)(ffn_output)
    
    return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

# Physics calculations
def remove_gravity_from_acc_vectorized(acc_values, quat_values):
    """Vectorized gravity removal for better performance"""
    num_samples = acc_values.shape[0]
    linear_accel = acc_values.copy()
    
    # Filter valid quaternions
    valid_mask = ~(np.any(np.isnan(quat_values), axis=1) | 
                   np.all(np.isclose(quat_values, 0), axis=1))
    
    if np.any(valid_mask):
        # Process all valid quaternions at once
        valid_quats = quat_values[valid_mask]
        
        # Batch rotation computation
        try:
            rotations = R.from_quat(valid_quats)
            gravity_world = np.array([0, 0, 9.81])
            
            # Apply rotations in batch
            gravity_sensor_frames = rotations.apply(gravity_world, inverse=True)
            linear_accel[valid_mask] = acc_values[valid_mask] - gravity_sensor_frames
        except:
            pass
            
    return linear_accel

def calculate_angular_velocity_vectorized(quat_values, time_delta=1/200):
    """Vectorized angular velocity calculation"""
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    
    if num_samples < 2:
        return angular_vel
        
    # Process in chunks for memory efficiency
    chunk_size = 1000
    for start in range(0, num_samples - 1, chunk_size):
        end = min(start + chunk_size, num_samples - 1)
        
        q_t = quat_values[start:end]
        q_t_plus_dt = quat_values[start+1:end+1]
        
        # Find valid pairs
        valid_mask = ~(np.any(np.isnan(q_t), axis=1) | 
                      np.all(np.isclose(q_t, 0), axis=1) |
                      np.any(np.isnan(q_t_plus_dt), axis=1) | 
                      np.all(np.isclose(q_t_plus_dt, 0), axis=1))
        
        if np.any(valid_mask):
            try:
                valid_indices = np.where(valid_mask)[0]
                rot_t = R.from_quat(q_t[valid_mask])
                rot_t_plus_dt = R.from_quat(q_t_plus_dt[valid_mask])
                
                # Batch computation
                delta_rot = rot_t.inv() * rot_t_plus_dt
                angular_vel[start + valid_indices] = delta_rot.as_rotvec() / time_delta
            except:
                pass
                
    return angular_vel

# Loss functions
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate focal loss
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        return K.mean(K.sum(loss, axis=1))
    
    return focal_loss_fixed

# Learning rate scheduler
def cosine_annealing_schedule(epoch, lr):
    """Cosine annealing with warmup"""
    warmup_epochs = 5
    max_epochs = 120
    
    if epoch < warmup_epochs:
        return LR_INIT * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
        return LR_INIT * 0.5 * (1 + np.cos(np.pi * progress))

# Model architecture
def build_enhanced_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)
    
    # IMU branch with transformer
    x1 = residual_se_cnn_block(imu, 64, 5, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)
    x1 = transformer_block(x1, dropout=0.2)
    
    # ToF branch with deeper architecture
    x2 = Conv1D(128, 1, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = residual_se_cnn_block(x2, 192, 3, drop=0.2, wd=wd)
    x2 = residual_se_cnn_block(x2, 256, 3, drop=0.2, wd=wd)
    x2 = transformer_block(x2, dropout=0.2)
    
    # Cross-attention fusion
    x1_to_x2 = MultiHeadAttention(num_heads=4, key_dim=64)(x1, x2)
    x2_to_x1 = MultiHeadAttention(num_heads=4, key_dim=64)(x2, x1)
    
    # Combine all features
    merged = Concatenate()([x1, x2, x1_to_x2, x2_to_x1])
    
    # Bidirectional LSTM with attention
    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd),
                          dropout=0.2, recurrent_dropout=0.2))(merged)
    
    # Multi-scale temporal pooling
    x_att = attention_layer(x)
    x_avg = GlobalAveragePooling1D()(x)
    x_max = GlobalMaxPooling1D()(x)
    
    x = Concatenate()([x_att, x_avg, x_max])
    
    # Enhanced classifier with residual connections
    x_res = x
    x = Dense(512, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, use_bias=False, kernel_regularizer=l2(wd))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Add skip connection
    x_res = Dense(256, kernel_regularizer=l2(wd))(x_res)
    x = Add()([x, x_res])
    
    # Output
    x = Dense(n_classes, kernel_regularizer=l2(wd), dtype='float32')(x)
    out = Activation('softmax', dtype='float32')(x)
    
    return Model(inp, out)

# Data augmentation
def augment_sequence(X, aug_prob=0.5):
    """Apply various augmentations to time series data"""
    if np.random.random() > aug_prob:
        return X
    
    aug_type = np.random.choice(['magnitude_warp', 'jitter', 'scaling'])
    
    if aug_type == 'magnitude_warp':
        scale_center = np.random.uniform(0.9, 1.1)
        scale_range = 0.1
        scales = np.random.uniform(scale_center - scale_range, 
                                  scale_center + scale_range, 
                                  size=(len(X), 1))
        return X * scales
    
    elif aug_type == 'jitter':
        noise = np.random.normal(0, 0.02, X.shape)
        return X + noise
    
    elif aug_type == 'scaling':
        scale = np.random.uniform(0.95, 1.05)
        return X * scale
    
    return X

# MixUp generator
class EnhancedMixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.4, augment_prob=0.3):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.augment_prob = augment_prob
        self.indices = np.arange(len(X))
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
        
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx].copy(), self.y[idx].copy()
        
        # Apply augmentation to some samples
        for j in range(len(Xb)):
            if np.random.random() < self.augment_prob:
                try:
                    Xb[j] = augment_sequence(Xb[j], aug_prob=1.0)
                except Exception as e:
                    print(f"Augmentation failed for sample {j}: {e}")
                    pass
        
        # MixUp
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        
        return X_mix.astype('float32'), y_mix.astype('float32')
        
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# Feature engineering functions
def add_frequency_features(seq_data, sampling_rate=200):
    """Add FFT-based features for key sensors"""
    seq_data = seq_data.copy()
    
    # Key columns for FFT
    fft_cols = ['acc_x', 'acc_y', 'acc_z', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z']
    
    for col in fft_cols:
        if col in seq_data.columns and len(seq_data) > 10:
            signal = seq_data[col].values
            
            # Compute FFT
            fft_vals = np.fft.fft(signal)
            fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
            
            # Get magnitude spectrum
            mag_spectrum = np.abs(fft_vals[:len(fft_vals)//2])
            freqs = fft_freq[:len(fft_freq)//2]
            
            if len(mag_spectrum) > 1:
                # Dominant frequency
                dom_freq_idx = np.argmax(mag_spectrum[1:]) + 1
                seq_data[f'{col}_dominant_freq'] = freqs[dom_freq_idx]
                seq_data[f'{col}_dominant_mag'] = mag_spectrum[dom_freq_idx]
                
                # Spectral energy in bands
                bands = [(0.5, 2), (2, 5), (5, 10), (10, 20), (20, 50)]
                for i, (low, high) in enumerate(bands):
                    mask = (freqs >= low) & (freqs < high)
                    if np.any(mask):
                        seq_data[f'{col}_energy_band{i}'] = np.sum(mag_spectrum[mask]**2)
                    else:
                        seq_data[f'{col}_energy_band{i}'] = 0
                
                # Spectral entropy
                mag_sum = np.sum(mag_spectrum)
                if mag_sum > 0:
                    norm_spectrum = mag_spectrum / mag_sum
                    norm_spectrum = norm_spectrum[norm_spectrum > 0]
                    spectral_entropy = -np.sum(norm_spectrum * np.log(norm_spectrum))
                    seq_data[f'{col}_spectral_entropy'] = spectral_entropy
                else:
                    seq_data[f'{col}_spectral_entropy'] = 0
    
    return seq_data

def add_cross_sensor_features(seq_data):
    """Add correlations between different sensor modalities"""
    seq_data = seq_data.copy()
    
    # IMU correlations
    if 'acc_mag' in seq_data.columns and 'angular_vel_z' in seq_data.columns:
        seq_data['acc_gyro_correlation'] = seq_data['acc_mag'].rolling(20, min_periods=5).corr(seq_data['angular_vel_z'])
    
    # ToF and motion correlation
    if 'tof_1_mean' in seq_data.columns and 'linear_acc_mag' in seq_data.columns:
        seq_data['tof_motion_corr'] = seq_data['tof_1_mean'].rolling(20, min_periods=5).corr(seq_data['linear_acc_mag'])
    
    # Thermopile and motion
    if 'thm_2' in seq_data.columns and 'acc_mag' in seq_data.columns:
        seq_data['thm_motion_corr'] = seq_data['thm_2'].rolling(20, min_periods=5).corr(seq_data['acc_mag'])
    
    # Fill NaN values from rolling operations
    seq_data = seq_data.fillna(0)
    
    return seq_data

# Process single sequence
def process_sequence(seq_data):
    """Process a single sequence with all feature engineering"""
    seq_data = seq_data.copy()
    
    # Dictionary to collect all new columns
    new_columns = {}
    
    # Get numpy arrays for faster processing
    acc_values = seq_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = seq_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values

    try:
        rotations = R.from_quat(quat_values)
        euler_angles = rotations.as_euler('xyz', degrees=True)
        new_columns['euler_x'] = euler_angles[:, 0]
        new_columns['euler_y'] = euler_angles[:, 1]
        new_columns['euler_z'] = euler_angles[:, 2]
    except:
        new_columns['euler_x'] = np.zeros(len(seq_data))
        new_columns['euler_y'] = np.zeros(len(seq_data))
        new_columns['euler_z'] = np.zeros(len(seq_data))
    
    # Linear acceleration
    linear_accel = remove_gravity_from_acc_vectorized(acc_values, quat_values)
    new_columns['linear_acc_x'] = linear_accel[:, 0]
    new_columns['linear_acc_y'] = linear_accel[:, 1]
    new_columns['linear_acc_z'] = linear_accel[:, 2]

    # Acceleration features
    new_columns['acc_yz_mag'] = np.sqrt(seq_data['acc_y']**2 + seq_data['acc_z']**2)
    new_columns['acc_y_z_ratio'] = seq_data['acc_y'] / (seq_data['acc_z'] + 1e-8)
    
    # Magnitudes
    new_columns['acc_mag'] = np.linalg.norm(acc_values, axis=1)
    new_columns['linear_acc_mag'] = np.linalg.norm(linear_accel, axis=1)
    
    # Jerk
    new_columns['linear_acc_mag_jerk'] = np.gradient(new_columns['linear_acc_mag']) * 200
    
    # Angular velocity
    angular_vel = calculate_angular_velocity_vectorized(quat_values)
    new_columns['angular_vel_x'] = angular_vel[:, 0]
    new_columns['angular_vel_y'] = angular_vel[:, 1]
    new_columns['angular_vel_z'] = angular_vel[:, 2]
    
    # ToF aggregations
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = seq_data[pixel_cols].values
        tof_data[tof_data == -1] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_columns[f'tof_{i}_mean'] = np.nanmean(tof_data, axis=1)
            new_columns[f'tof_{i}_std'] = np.nanstd(tof_data, axis=1)
            new_columns[f'tof_{i}_min'] = np.nanmin(tof_data, axis=1)
            new_columns[f'tof_{i}_max'] = np.nanmax(tof_data, axis=1)

    # Rotation features
    new_columns['rot_z_w_product'] = seq_data['rot_z'] * seq_data['rot_w']
    new_columns['rot_z_w_ratio'] = seq_data['rot_z'] / (seq_data['rot_w'] + 1e-8)

    # Rolling statistics
    for col in ['rot_z', 'rot_w', 'acc_z', 'acc_y', 'thm_2']:
        new_columns[f'{col}_rolling_mean'] = seq_data[col].rolling(10, center=True).mean()
        new_columns[f'{col}_rolling_std'] = seq_data[col].rolling(10, center=True).std()
        new_columns[f'{col}_diff'] = seq_data[col].diff()
    
    # Thermopile features
    thm_2_mean = seq_data['thm_2'].mean()
    thm_2_std = seq_data['thm_2'].std()
    new_columns['thm_2_normalized'] = (seq_data['thm_2'] - thm_2_mean) / (thm_2_std + 1e-8)
    new_columns['thm_2_delta_from_mean'] = seq_data['thm_2'] - seq_data[['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']].mean(axis=1)

    # ToF band statistics (simplified for brevity - includes all bands from notebook)
    # ToF 1 and 2 bands
    bands_tof_1_2 = [(3, 7), (11, 15), (19, 23), (27, 31), (35, 39), (43, 47), (51, 55), (59, 63)]
    for tof_num in [1, 2]:
        for band_idx, (start, end) in enumerate(bands_tof_1_2, 1):
            band_cols = [f"tof_{tof_num}_v{p}" for p in range(start, end + 1)]
            band_data = seq_data[band_cols].values
            band_data[band_data == -1] = np.nan
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                new_columns[f'tof_{tof_num}_band{band_idx}_mean'] = np.nanmean(band_data, axis=1)
                new_columns[f'tof_{tof_num}_band{band_idx}_std'] = np.nanstd(band_data, axis=1)
                new_columns[f'tof_{tof_num}_band{band_idx}_var'] = np.nanvar(band_data, axis=1)
                new_columns[f'tof_{tof_num}_band{band_idx}_min'] = np.nanmin(band_data, axis=1)
                new_columns[f'tof_{tof_num}_band{band_idx}_max'] = np.nanmax(band_data, axis=1)
    
    # ToF 3 bands
    bands_tof_3 = [(0, 47), (50, 55)]
    for band_idx, (start, end) in enumerate(bands_tof_3, 1):
        band_cols = [f"tof_3_v{p}" for p in range(start, end + 1)]
        band_data = seq_data[band_cols].values
        band_data[band_data == -1] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_columns[f'tof_3_band{band_idx}_mean'] = np.nanmean(band_data, axis=1)
            new_columns[f'tof_3_band{band_idx}_std'] = np.nanstd(band_data, axis=1)
            new_columns[f'tof_3_band{band_idx}_var'] = np.nanvar(band_data, axis=1)
            new_columns[f'tof_3_band{band_idx}_min'] = np.nanmin(band_data, axis=1)
            new_columns[f'tof_3_band{band_idx}_max'] = np.nanmax(band_data, axis=1)
    
    # ToF 4 bands
    bands_tof_4 = [(0, 3), (7, 9), (15, 16), (21, 23), (28, 31), (35, 39), (43, 47), (50, 55), (58, 63)]
    for band_idx, (start, end) in enumerate(bands_tof_4, 1):
        band_cols = [f"tof_4_v{p}" for p in range(start, end + 1)]
        band_data = seq_data[band_cols].values
        band_data[band_data == -1] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_columns[f'tof_4_band{band_idx}_mean'] = np.nanmean(band_data, axis=1)
            new_columns[f'tof_4_band{band_idx}_std'] = np.nanstd(band_data, axis=1)
            new_columns[f'tof_4_band{band_idx}_var'] = np.nanvar(band_data, axis=1)
            new_columns[f'tof_4_band{band_idx}_min'] = np.nanmin(band_data, axis=1)
            new_columns[f'tof_4_band{band_idx}_max'] = np.nanmax(band_data, axis=1)
    
    # ToF 5 bands
    bands_tof_5 = [(1, 7), (9, 15), (18, 23), (48, 49), (56, 61)]
    for band_idx, (start, end) in enumerate(bands_tof_5, 1):
        band_cols = [f"tof_5_v{p}" for p in range(start, end + 1)]
        band_data = seq_data[band_cols].values
        band_data[band_data == -1] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            new_columns[f'tof_5_band{band_idx}_mean'] = np.nanmean(band_data, axis=1)
            new_columns[f'tof_5_band{band_idx}_std'] = np.nanstd(band_data, axis=1)
            new_columns[f'tof_5_band{band_idx}_var'] = np.nanvar(band_data, axis=1)
            new_columns[f'tof_5_band{band_idx}_min'] = np.nanmin(band_data, axis=1)
            new_columns[f'tof_5_band{band_idx}_max'] = np.nanmax(band_data, axis=1)
    
    # Create DataFrame from new columns and concatenate
    new_columns_df = pd.DataFrame(new_columns, index=seq_data.index)
    result_df = pd.concat([seq_data, new_columns_df], axis=1)

    # Add frequency features
    result_df = add_frequency_features(result_df)
    
    # Add cross-sensor correlations
    result_df = add_cross_sensor_features(result_df)
    
    return result_df

def load_data(df):
    """Load and process data with parallel computation"""
    df = df.copy()
    
    print("Processing sequences with parallel computation...")
    
    # Group by sequence
    sequences = [group for _, group in df.groupby('sequence_id')]
    
    # Process in parallel
    n_cores = multiprocessing.cpu_count()
    print(f"Using {n_cores} cores for parallel processing")
    
    processed_sequences = Parallel(n_jobs=n_cores)(
        delayed(process_sequence)(seq) for seq in tqdm(sequences, desc="Processing sequences")
    )

    # Combine processed sequences
    df = pd.concat(processed_sequences, ignore_index=True)
    
    return df

def feature_engineering(df):
    """Additional feature engineering"""
    df = df.copy()

    # checks N/A before combining
    def combine_or_na(a, b):
        if 'N/A' in str(a) or 'N/A' in str(b):
            return 'N/A'
        return f"{a}_{b}"

    # combine orientation and gesture
    df['orientation_gesture'] = df.apply(lambda x: combine_or_na(x['orientation'], x['gesture']), axis=1).astype('category') 
        
    # behavioural boolean columns
    df['performs_gesture'] = df['behavior'].str.contains('Performs gesture', case=False, na=False)
    df['move_hand_to_target'] = df['behavior'].str.contains('Moves hand to target location', case=False, na=False)
    df['hand_at_target'] = df['behavior'].str.contains('Hand at target location', case=False, na=False)
    df['relaxes_moves_hand_to_target'] = df['behavior'].str.contains('Relaxes and moves hand to target location', case=False, na=False)
    
    return df

def fill_missing_values(df):
    """Fill missing values in dataframe"""
    # Fill categorical columns with 'N/A'
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna("N/A")
    
    # Fill numerical columns with 0
    num_cols = df.select_dtypes(include=[np.number]).columns  
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf, '', None], np.nan).fillna(0)
    
    return df

def main():
    """Main training function"""
    print("CMI Behavior Detection Training Script")
    print("=====================================")
    
    # Set random seed
    seed_everything(42)
    
    # Check for GPU
    print(f"\nGPU Available: {tf.config.list_physical_devices('GPU')}")
    
    # Load data
    print(f"\nLoading data from {RAW_DIR}")
    train_df = pd.read_csv(RAW_DIR / "train.csv")
    train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
    train_df = pd.merge(train_df, train_dem_df, on='subject', how='left')
    
    print(f"Data shape: {train_df.shape}")
    
    # Process data
    processed_df = load_data(train_df)
    fe_train_df = feature_engineering(processed_df)
    fe_train_df = fill_missing_values(fe_train_df)
    
    # Encode labels
    le = LabelEncoder()
    fe_train_df['gesture_int'] = le.fit_transform(fe_train_df['gesture'])
    gesture_classes = le.classes_
    np.save(OUTPUT_DIR / "gesture_classes.npy", gesture_classes)
    
    print(f"\nGesture classes: {gesture_classes}")
    
    # Handle categorical features
    cat_features = fe_train_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    encoder_mappings = {}
    
    print("\nLabel encoding categorical features...")
    for c in cat_features:
        fe_train_df[c] = fe_train_df[c].astype('category')
        full_le = LabelEncoder()
        fe_train_df[c] = full_le.fit_transform(fe_train_df[c].astype(str))
        encoder_mappings[c] = {i: label for i, label in enumerate(full_le.classes_)}
    
    with open(OUTPUT_DIR / 'categorical_encoder_mappings.json', 'w') as f:
        json.dump(encoder_mappings, f, indent=2)
    
    # Define feature columns
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
                'rot_x', 'rot_y', 'rot_z', 'rot_w', 'acc_mag', 'linear_acc_mag', 
                'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                'acc_y_rolling_mean', 'acc_y_rolling_std', 'acc_z_rolling_mean',
                'acc_z_rolling_std', 'rot_w_rolling_mean', 'rot_w_rolling_std',
                'rot_z_rolling_mean', 'rot_z_w_product', 'euler_x', 'euler_y', 'euler_z']
    
    thm_cols = ['thm_1', 'thm_2_delta_from_mean', 'thm_2_normalized', 'thm_2_rolling_std',
                'thm_2', 'thm_3', 'thm_4', 'thm_5']
    
    tof_aggregated_cols = []
    # Add ToF band features
    for tof_num in [1, 2]:
        for band_idx in range(1, 9):
            tof_aggregated_cols.append(f'tof_{tof_num}_band{band_idx}_mean')
    
    for band_idx in range(1, 3):
        tof_aggregated_cols.append(f'tof_3_band{band_idx}_mean')
    
    for band_idx in range(1, 10):
        tof_aggregated_cols.append(f'tof_4_band{band_idx}_mean')
    
    for band_idx in range(1, 6):
        tof_aggregated_cols.append(f'tof_5_band{band_idx}_mean')
    
    # Add ToF aggregations
    for i in range(1, 6):
        tof_aggregated_cols.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])
    
    final_feature_cols = imu_cols + thm_cols + tof_aggregated_cols
    imu_dim = len(imu_cols)
    tof_thm_dim = len(thm_cols) + len(tof_aggregated_cols)
    
    print(f"\nFeature dimensions - IMU: {imu_dim}, THM + ToF: {tof_thm_dim}")
    np.save(OUTPUT_DIR / "feature_cols.npy", np.array(final_feature_cols))
    
    # Build sequences
    print("\nBuilding sequences...")
    seq_gp = fe_train_df.groupby('sequence_id')
    X_list_unscaled = []
    y_list_int = []
    groups_list = []
    lens = []
    
    for seq_id, seq_df in seq_gp:
        X_list_unscaled.append(seq_df[final_feature_cols].fillna(0).values.astype('float32'))
        y_list_int.append(seq_df['gesture_int'].iloc[0])
        groups_list.append(seq_df['subject'].iloc[0])
        lens.append(len(seq_df))
    
    # Scaling
    print("Fitting StandardScaler...")
    all_steps_concatenated = np.concatenate(X_list_unscaled, axis=0)
    scaler = StandardScaler().fit(all_steps_concatenated)
    joblib.dump(scaler, OUTPUT_DIR / "scaler.pkl")
    
    # Scale and pad
    X_scaled_list = [scaler.transform(x_seq) for x_seq in X_list_unscaled]
    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(OUTPUT_DIR / "sequence_maxlen.npy", pad_len)
    
    X = pad_sequences(X_scaled_list, maxlen=pad_len, padding='post', 
                      truncating='post', dtype='float32')
    y_stratify = np.array(y_list_int)
    groups = np.array(groups_list)
    y = to_categorical(y_list_int, num_classes=len(le.classes_))
    
    print(f"\nFinal data shape: X={X.shape}, y={y.shape}")
    
    # Cross-validation training
    print(f"\nStarting {N_SPLITS}-fold cross-validation training...")
    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_stratify, groups)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold+1}/{N_SPLITS}")
        print(f"{'='*50}")
        
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        print(f"Train: {X_tr.shape}, Val: {X_val.shape}")
        
        # Create model
        model = build_enhanced_model(
            pad_len=pad_len, 
            imu_dim=imu_dim, 
            tof_dim=tof_thm_dim, 
            n_classes=len(le.classes_), 
            wd=WD
        )
        
        # Compile with mixed precision optimizer
        opt = Adam(learning_rate=LR_INIT)
        opt = mixed_precision.LossScaleOptimizer(opt)
        
        model.compile(
            optimizer=opt,
            loss=focal_loss(gamma=2.0, alpha=0.25),
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=PATIENCE, 
                restore_best_weights=True, 
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=10, 
                min_lr=1e-6, 
                verbose=1
            ),
            LearningRateScheduler(cosine_annealing_schedule, verbose=1),
            tf.keras.callbacks.ModelCheckpoint(
                str(OUTPUT_DIR / f'model_fold_{fold}_best.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        # Train
        train_gen = EnhancedMixupGenerator(
            X_tr, y_tr, 
            batch_size=BATCH_SIZE, 
            alpha=MIXUP_ALPHA,
            augment_prob=0.3
        )
        
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(OUTPUT_DIR / f"model_fold_{fold}_final.h5")
        
        # Clear session to free memory
        tf.keras.backend.clear_session()
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Models saved to: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()