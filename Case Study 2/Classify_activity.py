# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import scipy.stats
import pywt

sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
# Last row of training data for train/test split
train_end_index = 3511

# FEATURE EXTRACTION FUNCTIONS + HELPERS
def RMS(X):
    return np.sqrt(np.mean(X**2, axis=1))

# Mean autocorrelation of one channel and one sensor
def mean_autocorr(v):
    v = v - np.mean(v)
    corr = np.correlate(v, v, mode='full')
    corr = corr[corr.size // 2:]
    corr = corr / corr[0]
    return corr[1:].mean()

# Fast fourier transform, magnitude of first three components and FFT entropy
def fft_mag_entropy_axis(V):
    Xf = np.fft.rfft(V, axis=1)
    mag = np.abs(Xf)

    first3 = mag[:, 1:4]

    power = mag**2
    power_sum = np.sum(power, axis=1, keepdims=True)
    p = power / (power_sum + 1e-12)
    ent = -np.sum(p * np.log(p + 1e-12), axis=1)

    return first3, ent

def fft_features_3axes(X_3):
    N, T, _ = X_3.shape

    mag_sig = np.linalg.norm(X_3, axis=2)

    x = X_3[:, :, 0]
    y = X_3[:, :, 1]
    z = X_3[:, :, 2]

    x_f3, x_ent = fft_mag_entropy_axis(x)
    y_f3, y_ent = fft_mag_entropy_axis(y)
    z_f3, z_ent = fft_mag_entropy_axis(z)
    m_f3, m_ent = fft_mag_entropy_axis(mag_sig)

    feats = np.hstack([
        x_f3, x_ent[:, None],
        y_f3, y_ent[:, None],
        z_f3, z_ent[:, None],
        m_f3, m_ent[:, None],
    ])

    return feats

def coeff_features(c):
    sabs = np.sum(np.abs(c))
    energy = np.sum(c**2)
    p = np.abs(c) / (np.sum(np.abs(c)) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return sabs, entropy, energy

def wavelet_feats_1d(v, wavelet='db4', level=3):
    coeffs = pywt.wavedec(v, wavelet=wavelet, level=level)
    feats = []
    for c in coeffs:
        feats.extend(coeff_features(c))
    return feats

def wavelet_features_3axes(X_3, wavelet='db4', level=3):
    N, T, _ = X_3.shape
    out = []

    mag = np.linalg.norm(X_3, axis=2)

    for i in range(N):
        fx = wavelet_feats_1d(X_3[i,:,0], wavelet, level)
        fy = wavelet_feats_1d(X_3[i,:,1], wavelet, level)
        fz = wavelet_feats_1d(X_3[i,:,2], wavelet, level)
        fm = wavelet_feats_1d(mag[i],        wavelet, level)

        out.append(fx + fy + fz + fm)

    return np.array(out)


def extract_features(X):
    features = []
    accel = X[:,:,0:3]
    gyro = X[:,:,3:6]
    
    # Extract Signal Magnitude for each sensor
    accel_mag = np.linalg.norm(accel, axis=2)
    gyro_mag  = np.linalg.norm(gyro, axis=2)
    
    # Means of signals and of signal mags
    mean_feat = np.mean(X, axis=1)
    mean_accel_mag = np.mean(accel_mag, axis=1)[:, None]
    mean_gyro_mag = np.mean(gyro_mag, axis=1)[:, None]
    
    features.append(mean_feat)
    features.append(mean_accel_mag)
    features.append(mean_gyro_mag)
    
    # Median of signals and of signal mags
    med_feat = np.median(X, axis=1)
    med_accel_mag = np.median(accel_mag, axis=1)[:, None]
    med_gyro_mag = np.median(gyro_mag, axis=1)[:, None]
    
    features.append(med_feat)
    features.append(med_accel_mag)
    features.append(med_gyro_mag)
    
    # Variance of signals and mags
    var_feat = np.var(X, axis=1)
    var_accel_mag = np.var(accel_mag, axis=1)[:, None]
    var_gyro_mag = np.var(gyro_mag, axis=1)[:, None]
    
    features.append(var_feat)
    features.append(var_accel_mag)
    features.append(var_gyro_mag)
    
    # Std deviation of signals and of signal mags
    std_feat = np.std(X, axis=1)
    std_accel_mag = np.std(accel_mag, axis=1)[:, None]
    std_gyro_mag = np.std(gyro_mag, axis=1)[:, None]
    
    features.append(std_feat)
    features.append(std_accel_mag)
    features.append(std_gyro_mag)
    
    # Min of signals and of signal mags
    min_feat = np.min(X, axis=1)
    min_accel_mag = np.min(accel_mag, axis=1)[:, None]
    min_gyro_mag = np.min(gyro_mag, axis=1)[:, None]
    
    features.append(min_feat)
    features.append(min_accel_mag)
    features.append(min_gyro_mag)
    
    # Max of signals and of signal mags
    max_feat = np.max(X, axis=1)
    max_accel_mag = np.max(accel_mag, axis=1)[:, None]
    max_gyro_mag = np.max(gyro_mag, axis=1)[:, None]
    
    features.append(max_feat)
    features.append(max_accel_mag)
    features.append(max_gyro_mag)
    
    # IQR of signals and of signal mags
    iqr_feat = np.subtract(*np.percentile(X, [75, 25], axis=1))
    iqr_accel_mag = np.subtract(*np.percentile(accel_mag, [75, 25], axis=1))[:, None]
    iqr_gyro_mag = np.subtract(*np.percentile(gyro_mag, [75, 25], axis=1))[:, None]
    
    features.append(iqr_feat)
    features.append(iqr_accel_mag)
    features.append(iqr_gyro_mag)
    
    # Skewness of signals and of signal mags
    skew_feat = scipy.stats.skew(X, axis=1)
    skew_accel_mag = scipy.stats.skew(accel_mag, axis=1)[:, None]
    skew_gyro_mag = scipy.stats.skew(gyro_mag, axis=1)[:, None]
    
    features.append(skew_feat)
    features.append(skew_accel_mag)
    features.append(skew_gyro_mag)

    
    # RMS of features and signal mags
    rms_feat = RMS(X)
    rms_accel_mag = RMS(accel_mag)[:, None]
    rms_gyro_mag = RMS(gyro_mag)[:, None]
    
    features.append(rms_feat)
    features.append(rms_accel_mag)
    features.append(rms_gyro_mag)
    
    # Mean autocorrelation of each signal over 60hz sample for each row
    N = X.shape[0]

    out = np.zeros((N, 8))

    for i in range(N):
        out[i, 0] = mean_autocorr(accel[i, :, 0])   # ax
        out[i, 1] = mean_autocorr(accel[i, :, 1])   # ay
        out[i, 2] = mean_autocorr(accel[i, :, 2])   # az
        out[i, 3] = mean_autocorr(accel_mag[i])     # accel magnitude

        out[i, 4] = mean_autocorr(gyro[i, :, 0])    # gx
        out[i, 5] = mean_autocorr(gyro[i, :, 1])    # gy
        out[i, 6] = mean_autocorr(gyro[i, :, 2])    # gz
        out[i, 7] = mean_autocorr(gyro_mag[i])      # gyro magnitude
        
    features.append(out)
    
    # FFT component mag and spectral entroy per sensor and signal mag
    accel_fft_feats = fft_features_3axes(accel)
    gyro_fft_feats  = fft_features_3axes(gyro)
    
    fft_feats = np.hstack([accel_fft_feats, gyro_fft_feats])
    
    features.append(fft_feats)
    
    # Wavelet features
    accel_wav_feats = wavelet_features_3axes(accel)
    gyro_wav_feats  = wavelet_features_3axes(gyro)
    
    wavelet_feats = np.hstack([accel_wav_feats, gyro_wav_feats])
    
    features.append(wavelet_feats)
    
    return np.hstack(features)
    

def predict_test(train_data, train_labels, test_data):
    
    X_train_extracted = extract_features(train_data)
    
    X_test_extracted = extract_features(test_data)
    
    model = RandomForestClassifier(
        n_estimators=700,
        criterion='log_loss',
        max_depth=8,
        n_jobs=-1,
        class_weight='balanced',
        random_state=67
        )
    
    model.fit(X_train_extracted, train_labels)
    test_outputs = model.predict(X_test_extracted)
    
    return test_outputs

# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    # Load labels and training sensor data into 3-D array
    labels = np.loadtxt('Train_1/labels_train_1.csv', dtype='int')
    data_slice_0 = np.loadtxt('Train_1/' + sensor_names[0] + '_train_1.csv',
                              delimiter=',')
    data = np.empty((data_slice_0.shape[0], data_slice_0.shape[1],
                     len(sensor_names)))
    data[:, :, 0] = data_slice_0
    del data_slice_0
    for sensor_index in range(1, len(sensor_names)):
        data[:, :, sensor_index] = np.loadtxt(
            'Train_1/' + sensor_names[sensor_index] + '_train_1.csv', delimiter=',')

    # Split into training and test by row index. Do not use a random split as
    # rows are not independent!
    train_data = data[:train_end_index+1, :, :]
    train_labels = labels[:train_end_index+1]
    test_data = data[train_end_index+1:, :, :]
    test_labels = labels[train_end_index+1:]
    test_outputs = predict_test(train_data, train_labels, test_data)
    
    # Compute micro and macro-averaged F1 scores
    micro_f1 = f1_score(test_labels, test_outputs, average='micro')
    macro_f1 = f1_score(test_labels, test_outputs, average='macro')
    print(f'Micro-averaged F1 score: {micro_f1}')
    print(f'Macro-averaged F1 score: {macro_f1}')
    
    # Examine outputs compared to labels
    n_test = test_labels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(n_test), test_labels, 'b.')
    plt.xlabel('Time window')
    plt.ylabel('Target')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(n_test), test_outputs, 'r.')
    plt.xlabel('Time window')
    plt.ylabel('Output (predicted target)')
    plt.show()
    
