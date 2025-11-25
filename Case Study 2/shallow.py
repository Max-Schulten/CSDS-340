#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 12:37:15 2025

@author: maximilianschulten

Used to test and fine-tune a shallow classifier for HAR
"""
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pywt
from itertools import product
import scipy
from pprint import pprint

#%% Config
train_end_index = 3511
sensor_names = ['Acc_x', 'Acc_y', 'Acc_z', 'Gyr_x', 'Gyr_y', 'Gyr_z']
train_suffix = '_train_1.csv'

#%% Data Load

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


# %% Feature Extraction Protocol
# Root mean square
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
    # Real FFT along time axis
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
    accel_wav_feats = wavelet_features_3axes(accel)  # (N, D1)
    gyro_wav_feats  = wavelet_features_3axes(gyro)   # (N, D1)
    
    wavelet_feats = np.hstack([accel_wav_feats, gyro_wav_feats])  # (N, 2*D1)
    
    features.append(wavelet_feats)
    
    return np.hstack(features)
    
    
#%% Extract Features

X_tr = extract_features(train_data)
X_te = extract_features(test_data)

X = extract_features(data)
y = labels

#%% Test Classifier(s)
models = [
    (RandomForestClassifier(n_estimators=500, criterion='log_loss', max_depth = 3, random_state = 67), "RF"),
    (SVC(C=1, random_state=67), "SVC"),
    (LinearSVC(C=1, random_state=67), "Lin-SVM"),
    (KNeighborsClassifier(n_neighbors=10), "kNN"),
    (LogisticRegression(penalty='l1', C=1, solver='saga', max_iter=99999, random_state=67), "LR"),
    (GaussianNB(), "NB"),
    ]

classifiers = []
macro_f1s = []
micro_f1s = []
train_macro_f1s = []
train_micro_f1s = []

for model, name in models:
    if name in ["LR", "kNN", "SVC"]:
        modelo = make_pipeline(
            StandardScaler(),
            model
            )
    else: 
        modelo = model
    
    classifiers.append(name)
    modelo.fit(X_tr, train_labels)

    test_outputs = modelo.predict(X_te)
    train_outputs = modelo.predict(X_tr)
    
    micro_f1 = f1_score(test_labels, test_outputs, average='micro')
    macro_f1 = f1_score(test_labels, test_outputs, average='macro')
    macro_f1s.append(macro_f1)
    micro_f1s.append(micro_f1)
    
    tr_micro_f1 = f1_score(train_labels, train_outputs, average='micro')
    tr_macro_f1 = f1_score(train_labels, train_outputs, average='macro')
    train_macro_f1s.append(tr_macro_f1)
    train_micro_f1s.append(tr_micro_f1)
    
    
#%% Visualize results
x = np.arange(len(classifiers))
w = 0.35

fig, ax = plt.subplots()
b1 = ax.bar(x - w/2, macro_f1s, width=w, label="Test Macro F1", color='red', alpha=0.4)
b2 = ax.bar(x + w/2, micro_f1s, width=w, label="Test Micro F1", color='orange', alpha=0.4)

# Macro bars
for i, bar in enumerate(b1):
    h = bar.get_height()
    cx = bar.get_x() + bar.get_width()/3
    train_val = train_macro_f1s[i]

    ax.plot([cx - 0.1, cx + 0.1], [train_val, train_val], color='red', linewidth=2)

# Micro bars
for i, bar in enumerate(b2):
    h = bar.get_height()
    cx = bar.get_x() + bar.get_width()/3
    train_val = train_micro_f1s[i]

    ax.plot([cx - 0.1, cx + 0.1], [train_val, train_val], color='orange', linewidth=2)

ax.plot([0,0], [0,0], color='red', linewidth=2, label='Train Macro F1')
ax.plot([0,0], [0,0], color='orange', linewidth=2, label='Train Micro F1')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.set_xlabel("Classifier")
ax.set_ylabel("Score")
ax.set_ylim((0, 1.1))
ax.set_title("Untuned Classifier Performance on\n208 Extracted Features (Pre-Submission)")
ax.legend()
plt.show()


# TIME TO HP TUNE BEST CLASSIFIERS
#%% Custom HP Tuning Procedure For Contiguous Blocks of Data
class IdentityScaler():
    def fit_transform(self, X):
        return X
    def transform(self, X):
        return X


def make_block_folds(n_samples, n_blocks):
    idx = np.arange(n_samples)
    blocks = np.array_split(idx, n_blocks)
    folds = []
    for i in range(n_blocks):
        test_idx = blocks[i]
        train_idx = np.concatenate([b for j, b in enumerate(blocks) if j != i])
        folds.append((train_idx, test_idx))
    return folds

def blocked_rf_grid_search(X, y, classifier, param_grid, scaler=IdentityScaler(),  n_blocks=5, scoring="macro"):
    n_samples = y.shape[0]
    folds = make_block_folds(n_samples, n_blocks)

    keys = list(param_grid.keys())
    best_score = -np.inf
    best_params = None

    for vals in product(*param_grid.values()):
        params = dict(zip(keys, vals))
        scores = []
        
        print("Running Cross-val on:")
        pprint(params)

        for train_idx, test_idx in folds:
            clf = classifier(**params)
            X_tr = X[train_idx]
            y_tr = y[train_idx]
            
            X_te = X[test_idx]
            y_te = y[test_idx]
            
            X_tr = scaler.fit_transform(X_tr)
            
            X_te = scaler.transform(X_te)
                
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

            if scoring == "macro":
                s = f1_score(y_te, y_pred, average="macro")
            elif scoring == "micro":
                s = f1_score(y_te, y_pred, average="micro")
            else:
                s = clf.score(X_te, y_te)

            scores.append(s)
        
        mean_score = np.mean(scores)
        print("Mean Score: ", mean_score)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score

#%% Holdout Test Set Scoring (Extrapolation)

def test_on_holdout(clf, scaler=IdentityScaler()):
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)
    
    clf.fit(X_tr_scaled, train_labels)
    preds = clf.predict(X_te_scaled)
    train_preds = clf.predict(X_tr_scaled)
    
    print("Micro and Macro on Training Set: ",
          f1_score(train_labels, train_preds, average='micro'),
          f1_score(train_labels, train_preds, average='macro')
          )

    print(
          f1_score(test_labels, preds, average='micro'),
          f1_score(test_labels, preds, average='macro')
          )

#%% Tuning RF

param_grid = {
    "n_estimators": [700],
    "max_depth": [8],
    "criterion": ['log_loss'],
    "random_state": [67],
    "n_jobs": [-1],
    "class_weight": ['balanced'],
    "max_features": ['log2', 'sqrt', 0.1, 0.2, 0.3, 0.4, 0.5]
    }

clf = RandomForestClassifier
best_params, best_score = blocked_rf_grid_search(X_tr, train_labels, clf, param_grid, scoring='micro')
print("Best mean score: ", best_score)
pprint(best_params)

test_on_holdout(RandomForestClassifier(**best_params))


 # %% Tuning LR
param_grid = {
    'penalty': ['l1'],
    'solver': ['saga'],
    'C': [1e-2],
    'max_iter': [999],
    'random_state': [67]
    }

clf = LogisticRegression
best_params, best_score = blocked_rf_grid_search(X_tr, train_labels, clf, param_grid, scoring='micro', scaler=StandardScaler())
print("Best mean score: ", best_score)
pprint(best_params)

test_on_holdout(LogisticRegression(**best_params), scaler=StandardScaler())

#%% Tuning KNN
param_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15],
    "weights": ['distance', 'uniform'],
    "metric": ['minkowski', 'cosine', 'euclidean', 'cityblock', 'chebyshev'],
    }

clf = KNeighborsClassifier
best_params, best_score = blocked_rf_grid_search(X_tr, train_labels, clf, param_grid, scoring='micro', scaler=StandardScaler())
print("Best mean score: ", best_score)
pprint(best_params)

test_on_holdout(KNeighborsClassifier(**best_params), scaler=StandardScaler())

#%% Tuning SVM
param_grid ={
    "C": [1e2,1e3,1e4],
    "class_weight": ['balanced', None],
    "gamma": [0.002],
    "random_state": [67]
    }

clf = SVC
best_params, best_score = blocked_rf_grid_search(X_tr, train_labels, clf, param_grid, scoring='micro', scaler=StandardScaler())
print("Best mean score: ", best_score)
pprint(best_params)

test_on_holdout(SVC(**best_params), scaler=StandardScaler())

