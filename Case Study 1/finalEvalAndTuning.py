#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model comparison on spam dataset
Evaluates tuned models via 5-fold CV and basic learning-curve diagnostics.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import roc_curve, make_scorer
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier


# ----- config ---------------------------------------------------------
RS = 11234
CV_FOLDS = 5
TUNING = False

data = np.vstack((np.loadtxt("spamTrain1.csv", delimiter=","),
                  np.loadtxt("spamTrain2.csv", delimiter=",")))
X, y = data[:, :-1], data[:, -1]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state = RS)


# ---- custom metric ---------------------------------------------------
def tpr_at_fpr_1perc(y_true, y_pred_proba, **kwargs):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    idx = np.where(fpr <= 0.01)[0]
    return 0.0 if len(idx) == 0 else float(tpr[idx[-1]])

tpr_at_1_fpr_scorer = make_scorer(
    tpr_at_fpr_1perc, response_method="predict_proba", greater_is_better=True
)

# ---- models ----------------------------------------------------------
pipe_xgb = make_pipeline(
    #SimpleImputer(missing_values=-1, strategy="mean"),
    XGBClassifier(
        objective="binary:logistic", eval_metric="auc",
        n_estimators=2000,
        learning_rate=0.003,
        max_depth=5,
        min_child_weight=1,
        gamma=1,
        subsample=0.7,
        colsample_bytree=0.5333,
        reg_alpha=0.1841,
        reg_lambda=0.5957,
        scale_pos_weight=0.9,
        max_delta_step=2,
        n_jobs=-1,
        random_state=67,
        missing=-1
    )
)

pipe_rf = make_pipeline(
    KNNImputer(missing_values=-1, n_neighbors=13, weights="distance"),
    RandomForestClassifier(
        n_estimators=1002, max_depth=14,
        min_samples_split=14, min_samples_leaf=5,
        class_weight={0: 1, 1: 2.5}, oob_score=True,
        n_jobs=-1, random_state=2834, bootstrap=True
    )
)

pipe_bagged_knn = make_pipeline(
    SimpleImputer(missing_values=-1),
    StandardScaler(),
    BaggingClassifier(
        estimator=KNeighborsClassifier(
            n_neighbors=12, metric="cosine", weights="uniform"
        ),
        n_estimators=1058, max_samples=0.6820016278344457,
        max_features=0.1871344621226954,
        bootstrap=True, bootstrap_features=True,
        oob_score=True, n_jobs=-1, random_state=RS
    )
)

models = [
    ("XGBoost", pipe_xgb),
    #("RandomForest", pipe_rf),
    #("Bagged_kNN", pipe_bagged_knn)
]

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RS)
scoring = {"roc_auc": "roc_auc", "tpr": tpr_at_1_fpr_scorer}

# ---------------------------------------------------------------------
def evaluate_model(name, model, ax):
    """
    Plot ROC-AUC and TPR@1%FPR learning curves for a single model on one subplot.
    Left y-axis: ROC-AUC (train & CV). Right y-axis: TPR (train & CV).
    """
    print(f"\n=== {name} ({CV_FOLDS}-fold CV) ===")
    cv_results = cross_validate(
        model, X, y, scoring=scoring, cv=cv, n_jobs=-1, return_train_score=True
    )

    for key in ["roc_auc", "tpr"]:
        mean = np.mean(cv_results[f"test_{key}"])
        std  = np.std(cv_results[f"test_{key}"])
        print(f"{key:20s}: {mean:.4f} ± {std:.4f}")

    # Use the same train_sizes grid for both metrics
    train_sizes = np.linspace(0.1, 1.0, 6)

    # --- Learning curve: ROC-AUC ---
    ts_auc, train_auc, test_auc = learning_curve(
        model, X, y, cv=cv, scoring="roc_auc",
        train_sizes=train_sizes, n_jobs=-1
    )
    train_auc_mean, test_auc_mean = np.mean(train_auc, axis=1), np.mean(test_auc, axis=1)
    gap_auc = train_auc_mean - test_auc_mean
    print(f"Generalization gap (mean train−test ROC-AUC): {np.mean(gap_auc):.4f}")

    # --- Learning curve: TPR@1%FPR (your custom 'tpr' scorer) ---
    ts_tpr, train_tpr, test_tpr = learning_curve(
        model, X, y, cv=cv, scoring=tpr_at_1_fpr_scorer,
        train_sizes=train_sizes, n_jobs=-1
    )
    train_tpr_mean, test_tpr_mean = np.mean(train_tpr, axis=1), np.mean(test_tpr, axis=1)
    gap_tpr = train_tpr_mean - test_tpr_mean
    print(f"Generalization gap (mean train−test TPR): {np.mean(gap_tpr):.4f}")

    # --- Plot on one subplot with twin y-axes ---
    # Left axis: ROC-AUC
    ax.plot(ts_auc, train_auc_mean, "o-", label="Train ROC-AUC")
    ax.plot(ts_auc, test_auc_mean,  "o--", label="CV ROC-AUC")
    ax.set_title(f"{name}, {CV_FOLDS}-Fold Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("ROC-AUC")
    ax.grid(True, alpha=0.3)

    # Right axis: TPR
    ax2 = ax.twinx()
    ax2.plot(ts_tpr, train_tpr_mean, "s-", label="Train TPR")
    ax2.plot(ts_tpr, test_tpr_mean,  "s--", label="CV TPR")
    ax2.set_ylabel("TPR @ 1% FPR")

    # Build a combined legend
    lines_left, labels_left = ax.get_legend_handles_labels()
    lines_right, labels_right = ax2.get_legend_handles_labels()
    ax.legend(lines_left + lines_right, labels_left + labels_right, loc="center left")

if not TUNING:
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 4), sharex=True)
    if len(models) == 1:
        axes = [axes]
    
    out = {}
    for ax, (name, model) in zip(axes, models):
        out[name] = evaluate_model(name, model, ax)
    
    plt.tight_layout()
    plt.show()
else:
    
    # Hotter
    param_dist_A = {
        "xgbclassifier__max_depth": [5, 6],
        "xgbclassifier__min_child_weight": [1, 2, 3, 5],
        "xgbclassifier__learning_rate": np.logspace(np.log10(0.015), np.log10(0.035), 6),
        "xgbclassifier__n_estimators": [1500, 2000, 2400],
        "xgbclassifier__gamma": [1.0, 1.5, 2.0],
        "xgbclassifier__subsample": np.linspace(0.55, 0.75, 5),
        "xgbclassifier__colsample_bytree": np.linspace(0.5, 0.7, 5),
        "xgbclassifier__reg_alpha": np.logspace(np.log10(0.05), np.log10(0.6), 6),
        "xgbclassifier__reg_lambda": np.logspace(np.log10(0.05), np.log10(1.5), 6),
        "xgbclassifier__scale_pos_weight": [1.0, 1.2, 1.5, 1.8],
        "xgbclassifier__max_delta_step": [0, 1, 2, 3]
    }
    
    # Zoom in on winner of A
    param_dist_zoomA = {
        "xgbclassifier__max_depth": [5, 6],
        "xgbclassifier__min_child_weight": [1, 2, 3],
        "xgbclassifier__learning_rate": np.logspace(np.log10(0.012), np.log10(0.020), 6),
        "xgbclassifier__n_estimators": [1300, 1500, 1700],
        "xgbclassifier__gamma": [0.8, 1.0, 1.2],
        "xgbclassifier__subsample": np.linspace(0.65, 0.75, 5),
        "xgbclassifier__colsample_bytree": np.linspace(0.5, 0.6, 4),
        "xgbclassifier__reg_alpha": np.logspace(np.log10(0.12), np.log10(0.35), 6),
        "xgbclassifier__reg_lambda": np.logspace(np.log10(0.5), np.log10(1.2), 6),
        "xgbclassifier__scale_pos_weight": [0.9, 1.0, 1.1],
        "xgbclassifier__max_delta_step": [2, 3],
    }

    # Cooler
    param_dist_B = {
        "xgbclassifier__max_depth": [4, 5],
        "xgbclassifier__min_child_weight": [2, 3, 5, 7, 10],
        "xgbclassifier__learning_rate": np.logspace(np.log10(0.006), np.log10(0.014), 6),
        "xgbclassifier__n_estimators": [1800, 2200, 2600],
        "xgbclassifier__gamma": [1.0, 1.5, 2.0],
        "xgbclassifier__subsample": np.linspace(0.60, 0.80, 5),
        "xgbclassifier__colsample_bytree": np.linspace(0.55, 0.80, 6),
        "xgbclassifier__reg_alpha": np.logspace(np.log10(0.05), np.log10(0.8), 6),
        "xgbclassifier__reg_lambda": np.logspace(np.log10(0.1), np.log10(2.0), 6),
        "xgbclassifier__scale_pos_weight": [0.8, 1.0, 1.2, 1.5],
        "xgbclassifier__max_delta_step": [0, 1, 2, 3]
    }


    
    # multi-metric so we can refit on TPR but also see ROC-AUC
    scoring_multi = {"tpr": tpr_at_1_fpr_scorer, "roc_auc": "roc_auc"}
    
    search = RandomizedSearchCV(
        estimator=pipe_xgb,
        param_distributions=param_dist_zoomA,
        n_iter=60,
        scoring=scoring_multi,
        refit="tpr",
        cv=cv,
        random_state=RS,
        n_jobs=-1,
        return_train_score=False
    )
    
    search.fit(X, y)
    
    print("\n=== RandomizedSearchCV (XGBoost, regularization-focused) ===")
    print("Best params:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    
    best_idx = search.best_index_
    print(f"\nBest CV TPR@1%FPR: {search.cv_results_['mean_test_tpr'][best_idx]:.4f} "
          f"± {search.cv_results_['std_test_tpr'][best_idx]:.4f}")
    print(f"Mean CV ROC-AUC at best TPR: {search.cv_results_['mean_test_roc_auc'][best_idx]:.4f} "
          f"± {search.cv_results_['std_test_roc_auc'][best_idx]:.4f}")
    
    best_xgb = search.best_estimator_
