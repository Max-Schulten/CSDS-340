#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomized CV searches for Bagged kNN, Random Forest, and XGBoost
with custom TPR@1% FPR metric (as refit) and ROC AUC reporting.
"""
import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.metrics import make_scorer, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

# -------------------------
# Config
# -------------------------
ITERATIONS = 50
RANDOM_STATE = 420
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# -------------------------
# Custom scoring: TPR@1% FPR
# -------------------------
def tpr_at_fpr_1perc(y_true, y_pred_proba, **kwargs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    idx = np.where(fpr <= 0.01)[0]
    if len(idx) == 0:
        return 0.0
    return float(tpr[idx[-1]])

tpr_at_1_fpr_scorer = make_scorer(
    tpr_at_fpr_1perc,
    response_method="predict_proba",
    greater_is_better=True
)

scoring = {"tpr": tpr_at_1_fpr_scorer, "roc_auc": "roc_auc"}

# -------------------------
# Data
# -------------------------
data = np.loadtxt("spamTrain1.csv", delimiter=",")
X = data[:, :-1]
y = data[:, -1]

# -------------------------
# Pipelines
# -------------------------
bagged_knn = make_pipeline(
    SimpleImputer(missing_values=-1, strategy="median"),
    StandardScaler(),
    BaggingClassifier(
        estimator=KNeighborsClassifier(n_neighbors=3, metric="minkowski", p=2),
        n_estimators=500,
        n_jobs=-1,
        max_features=4, 
        max_samples=0.5,
        random_state=RANDOM_STATE,
    ),
)

RF = make_pipeline(
    SimpleImputer(missing_values=-1, strategy="median"),
    RandomForestClassifier(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_split=10,
        max_features="sqrt",
        max_depth=5,
        class_weight={0: 1, 1: 2},
        n_jobs=-1,
    ),
)

xgb = make_pipeline(
    SimpleImputer(missing_values=-1, strategy="median"),
    XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=500,
        learning_rate=0.01,
        max_depth=2,
        reg_alpha=1,
        reg_lambda=2,
        gamma=1,
        n_jobs=-1,
        enable_categorical=False,
        eval_metric="logloss",
    ),
)

# -------------------------
# Parameter distributions
# -------------------------
param_dist_bag_knn = {
    "baggingclassifier__n_estimators": randint(300, 800),
    "baggingclassifier__max_samples": uniform(0.1, 0.7),
    "baggingclassifier__max_features": uniform(0.1, 0.9),
    "baggingclassifier__bootstrap": [True],
    "baggingclassifier__bootstrap_features": [False],
    "baggingclassifier__estimator__n_neighbors": randint(1, 31),
    "baggingclassifier__estimator__weights": ["uniform", "distance"],
    "baggingclassifier__estimator__p": [1, 2],
}

param_dist_rf = {
    "randomforestclassifier__n_estimators": randint(200, 1200),
    "randomforestclassifier__max_depth": randint(3, 20),
    "randomforestclassifier__min_samples_split": randint(2, 20),
    "randomforestclassifier__min_samples_leaf": randint(1, 10),
    "randomforestclassifier__max_features": ["sqrt", "log2", None],
    "randomforestclassifier__class_weight": [
        None,
        "balanced",
        "balanced_subsample",
        {0: 1, 1: 2},
        {0: 1, 1: 3},
    ],
}

param_dist_xgb = {
    "xgbclassifier__n_estimators": randint(200, 1200),
    "xgbclassifier__max_depth": randint(2, 8),
    "xgbclassifier__learning_rate": loguniform(1e-3, 3e-1),
    "xgbclassifier__subsample": uniform(0.5, 0.5),         # (0.5, 1.0]
    "xgbclassifier__colsample_bytree": uniform(0.5, 0.5),  # (0.5, 1.0]
    "xgbclassifier__min_child_weight": randint(1, 10),
    "xgbclassifier__reg_alpha": loguniform(1e-3, 10),
    "xgbclassifier__reg_lambda": loguniform(1e-3, 10),
    "xgbclassifier__gamma": loguniform(1e-3, 1),
}

# -------------------------
# Randomized searches
# -------------------------
def run_search(name, estimator, param_dist):
    rs = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=ITERATIONS,
        scoring=scoring,
        refit="tpr",
        cv=CV,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )
    rs.fit(X, y)
    best_idx = rs.best_index_
    best_tpr = rs.best_score_
    best_roc = rs.cv_results_["mean_test_roc_auc"][best_idx]
    print("\n" + "=" * 80)
    print(f"{name}")
    print("-" * 80)
    print("Best params:")
    for k, v in rs.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV TPR@1% FPR: {best_tpr:.4f}")
    print(f"ROC AUC at best TPR point: {best_roc:.4f}")
    return name, best_tpr, best_roc, rs

results = []
results.append(run_search("Bagged kNN", bagged_knn, param_dist_bag_knn))
#results.append(run_search("Random Forest", RF, param_dist_rf))
#results.append(run_search("XGBoost", xgb, param_dist_xgb))

# -------------------------
# Leaderboard
# -------------------------
print("\n" + "#" * 80)
print("Leaderboard (by CV TPR@1% FPR)")
print("#" * 80)
for name, best_tpr, best_roc, _ in sorted(results, key=lambda t: t[1], reverse=True):
    print(f"{name:15s} | TPR@1%FPR: {best_tpr:7.4f} | ROC AUC: {best_roc:7.4f}")
