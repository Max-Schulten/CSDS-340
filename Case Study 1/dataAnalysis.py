#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 21:57:46 2025

@author: maximilianschulten
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from scipy.stats import shapiro
from sklearn.impute import SimpleImputer


data = np.loadtxt('spamTrain1.csv',delimiter=',')
# Separate labels (last column)
X = data[:,:-1]
y = data[:,-1]
feature_names = [f'{i}' for i in range(X.shape[1])]

"""
# ---- Train Random Forest ----
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf.fit(X, y)

# ---- Get Importances ----
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]  # sort descending
mean_importance = np.mean(importances)
median_importance = np.median(importances)

# ---- Plot ----
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(30), importances[indices][:30], align="center")
plt.xticks(range(30), [feature_names[i] for i in indices[:30]], rotation=0, ha="center")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.axhline(mean_importance, color='red', linestyle='--', linewidth=2,
            label=f"Mean Importance = {mean_importance:.4f}")
plt.axhline(median_importance, color='orange', linestyle='-', linewidth=2,
            label=f"Median Importance = {mean_importance:.4f}")
plt.show()

"""
# Check for class balance
print("--Class Frequencies--")
print(f"Class 1 (Spam): n={(y==1).sum()}\nClass 0 (Not Spam): n={(y==0).sum()}")
print('-'*50)
# Pandas dataframe easier to work with
df = pd.DataFrame(X)
# Nan's so pandas recognizes them as missing
df = df.replace(-1, np.nan)
# Compute # of missing values
missingness = df.isna().sum(axis=0)
print("--Overall Missingness and Normality--")
print(f"Average Missingness: {missingness.mean():.3f} +/- {missingness.std():.3f}")
print(f"Complete Rows: {df.dropna().shape[0]}")

# Perform shapiro-wilkes test for normality on each column
cols = df.columns
p_vals = []
for col in cols:
    clean_col = df[col].dropna()
    stat, p = shapiro(clean_col)
    p_vals.append(p)
p_vals = np.array(p_vals)

print(f"{(p_vals > 0.05).sum()} Columns Possibly Normally Distributed")

print('-'*50)
# Computer # of missing values per class, normality test per class per column
print("--Per Class Missingness and Normality--")
for label in range(0,2):
    mask = y == label
    df_masked = df[mask]
    missingness = df_masked.isna().sum(axis=0)
    print(f"Average Missingness % (Class {label}): {missingness.mean()/df_masked.shape[0]:.3f} +/- {missingness.std()/df_masked.shape[0]:.3f}")
    cols = df_masked.columns
    p_vals = []
    for col in cols:
        stat,p = 0,0
        clean_col = df_masked[col].dropna()
        if clean_col.nunique() > 1:
            stat, p = shapiro(clean_col)
        else:
            p = 0
        p_vals.append(p)
    p_vals = np.array(p_vals)

    print(f"{(p_vals > 0.05).sum()} Columns Possibly Normally Distributed")
print('-'*50)

# Correlational analysis
print('--Correlation Between Features--')
corr = df.corr().abs()
high_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
print(high_corr[high_corr > 0.5].stack())
print('-'*50)

# Variance analysis
print('--Variance of Features--')
variances = df.var()
print(f'Mean Variance After Scaling: {variances.mean():.3f} +/- {variances.std():.3f}')
print('-'*50)

# Feature importance analysis
rf = RandomForestClassifier(n_jobs=1,random_state=1)
X_imp = SimpleImputer(missing_values=-1, strategy='median').fit_transform(X)
rf.fit(X_imp, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]  # sort descending
mean_importance = np.mean(importances)
median_importance = np.median(importances)

X_scaled = StandardScaler().fit_transform(X_imp)
lr = LogisticRegression(C=0.1, 
                        penalty='l1', 
                        solver='saga', 
                        random_state=1,
                        max_iter = 9999,
                        class_weight={0:1, 1:2}).fit(X_scaled, y)
lr_weights_importance = lr.coef_ > 0
kept_features_LR = np.where(lr.coef_ > 0)
kept_features_rf = np.where(importances > mean_importance)
print(f"LR keeps {lr_weights_importance.sum()} feature(s)")
print(f"RF keeps {lr_weights_importance.sum()} feature(s)")

LR_rf_agree = np.intersect1d(kept_features_LR, kept_features_rf)


plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(30), importances[indices][:30], align="center")
plt.xticks(range(30), [feature_names[i] for i in indices[:30]], rotation=0, ha="center")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.axhline(mean_importance, color='red', linestyle='--', linewidth=2,
            label=f"Mean Importance = {mean_importance:.4f}")
plt.axhline(median_importance, color='orange', linestyle='-', linewidth=2,
            label=f"Median Importance = {mean_importance:.4f}")
plt.show()
# Plot with seaborn
"""
missing_counts = df.isna().sum().reset_index()
missing_counts.columns = ["feature", "missing_count"]
plt.figure(figsize=(12, 6))
sns.barplot(data=missing_counts, x="feature", y="missing_count")
plt.xticks(rotation=0, ha="center")
plt.title("Missingness by Feature")
plt.ylabel("Number of Missing Values")
plt.xlabel("Feature")
plt.tight_layout()
plt.show()
"""


# Density plots of each feature.
"""
df_vis = df.copy()
df_vis["label"] = y

for col in df_vis.columns[:-1]:
    plt.figure(figsize=(7,4))
    sns.kdeplot(data=df_vis, x=col, hue="label", common_norm=False)
    plt.title(f"Distribution of Feature {col} by Class")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()
"""