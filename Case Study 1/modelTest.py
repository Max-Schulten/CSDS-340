#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 11:24:01 2025

@author: maximilianschulten
"""
import numpy as np
from sklearn.metrics import make_scorer, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ITERATIONS = 20


# -- CUSTOM SCORING -- 
def tpr_at_fpr_1perc(y_true, y_pred_proba, **kwargs):
    # Predict probabilities
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    # Find closest index to desired FPR
    idx = np.where(fpr <= 0.01)[0]
    if len(idx) == 0:  # No FPR below target → return 0
        return 0.0

    # Take the last valid TPR under the target FPR
    return tpr[idx[-1]]

tpr_at_1_fpr_scorer = make_scorer(tpr_at_fpr_1perc, response_method='predict_proba', greater_is_better=True)

# -- DATA --
data = np.loadtxt('spamTrain1.csv',delimiter=',')
# Separate labels (last column)
X = data[:,:-1]
y = data[:,-1]

bagged_NB = make_pipeline(
    SimpleImputer(missing_values=-1, strategy='median'),
    BaggingClassifier(
            estimator=GaussianNB(),
            n_estimators=500,
            n_jobs=-1,
            max_features=2,
            max_samples=50,
            random_state=1
        )
    )

RF = make_pipeline(
    SimpleImputer(missing_values=-1, strategy='median'),
    RandomForestClassifier(
        n_estimators=500,
        random_state=1,
        min_samples_split=10,
        max_features='sqrt',
        max_depth=5,
        class_weight={0:1, 1:2}
        )
    )



bagged_knn = make_pipeline(
    SimpleImputer(missing_values=-1, strategy='median'),
    StandardScaler(),
    BaggingClassifier(
            estimator=KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=2),
            n_estimators=500,
            n_jobs=-1,
            max_features=4,
            max_samples=0.5,
            random_state=1
        )
    )


xgb = make_pipeline(
    SimpleImputer(missing_values=-1, strategy='median'),
    StandardScaler(),
    XGBClassifier(
            random_state=1,
            n_estimators=500,
            learning_rate=0.01,
            max_depth=2,
            reg_alpha=1,
            reg_lambda=2,
            gamma=1
        )
    )



train_size_abs, train_scores, test_scores = learning_curve(bagged_knn, X, y, 
                                                           train_sizes=[0.3, 0.6, 0.9],
                                                           scoring=tpr_at_1_fpr_scorer)

for train_size, cv_train_scores, cv_test_scores\
    in zip(train_size_abs, train_scores, test_scores):

    print(f"{train_size} samples were used to train the model")

    print(f"The average train accuracy is {cv_train_scores.mean():.2f}")

    print(f"The average test accuracy is {cv_test_scores.mean():.2f}")
    
    
print("#"*50)
train_size_abs, train_scores, test_scores = learning_curve(RF, X, y, 
                                                           train_sizes=[0.3, 0.6, 0.9],
                                                           scoring=tpr_at_1_fpr_scorer)

for train_size, cv_train_scores, cv_test_scores\
    in zip(train_size_abs, train_scores, test_scores):

    print(f"{train_size} samples were used to train the model")

    print(f"The average train accuracy is {cv_train_scores.mean():.2f}")

    print(f"The average test accuracy is {cv_test_scores.mean():.2f}")
