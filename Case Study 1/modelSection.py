import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA

# Evaluate TPR @ FPR = 0.01 under Cross-Validation
def tpr_at_fpr_cv(estimator, X, y, fpr_target=0.01, cv = None):
    cv = cv if cv else StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    tprs = []

    for train_idx, test_idx in cv.split(X, y):
        est = clone(estimator)
        est.fit(X[train_idx], y[train_idx])

        # Compute scores for ROC
        scores = est.predict_proba(X[test_idx])[:, 1]

        # Compute ROC
        fpr, tpr, _ = roc_curve(y[test_idx], scores)
        
        maxFprIndex = np.where(fpr<=fpr_target)[0][-1]
        fprBelow = fpr[maxFprIndex]
        fprAbove = fpr[maxFprIndex+1]
        # Find TPR at exactly desired FPR by linear interpolation
        tprBelow = tpr[maxFprIndex]
        tprAbove = tpr[maxFprIndex+1]
        tprAt = ((tprAbove-tprBelow)/(fprAbove-fprBelow)*(fpr_target-fprBelow) 
                 + tprBelow)
        tprs.append(tprAt)

    return np.mean(tprs), np.std(tprs)


# Load data
data = np.loadtxt('spamTrain1.csv',delimiter=',')
# Separate labels (last column)
X = data[:,:-1]
y = data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=1/3, 
                                                    random_state=200,
                                                    stratify=y)

# Define CV strategy
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)


# Build pipeline
classifiers = {
    "Med. Imp, PCA, KNN": make_pipeline(SimpleImputer(strategy='median',
                                                      missing_values=-1),
                                        StandardScaler(),
                                        PCA(n_components=10, random_state=1),
                                        KNeighborsClassifier(n_neighbors=10,
                                                             p=1)),
    "Knn Imp, PCA, KNN": make_pipeline(KNNImputer(n_neighbors=15, 
                                                   missing_values=-1),
                                        StandardScaler(),
                                        PCA(n_components=10, random_state=1),
                                        KNeighborsClassifier(n_neighbors=15,
                                                             p=1)),
    "Med. Imp, LR(L1) FS, LR(L2)": make_pipeline(SimpleImputer(strategy='median',
                                                              missing_values=-1),
                                                StandardScaler(),
                                                SelectFromModel(LogisticRegression(C=0.1,
                                                                                   penalty='l1',
                                                                                   solver='saga',
                                                                                   random_state=1,
                                                                                   class_weight={0:1, 1:2},
                                                                                   max_iter=9999)),
                                                LogisticRegression(class_weight={0:1, 1:2},
                                                                   random_state=1,
                                                                   max_iter=9999)),
    "Med. Imp, RF": make_pipeline(SimpleImputer(strategy='median'),
                                  RandomForestClassifier(n_estimators=300,
                                                         class_weight={0:1, 1:2},
                                                         n_jobs=-1,
                                                         random_state=1)),
    "Knn Imp, RF": make_pipeline(KNNImputer(n_neighbors=15, 
                                                   missing_values=-1),
                                  RandomForestClassifier(n_estimators=300,
                                                         class_weight={0:1, 1:2},
                                                         n_jobs=-1,
                                                         random_state=1)),
    "Med. Imp, RF FS, SVC": make_pipeline(SimpleImputer(strategy='median',
                                                        missing_values=-1),
                                          SelectFromModel(RandomForestClassifier(n_estimators=300,
                                                                                 n_jobs=-1,
                                                                                 random_state=1)),
                                          SVC(class_weight={0:1, 1:2},
                                              probability=True,
                                              random_state=1,
                                              C=0.01))
 }

for name, classifier in classifiers.items():
    
    classifier
    
    auc_scores = cross_val_score(
        classifier, X, y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    
    tprMean, tprStd = tpr_at_fpr_cv(classifier, X, y, cv=cv )
    
    print(f"Classifier: {name}, Mean AUC {cv.n_splits}-fold CV: {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}, Mean TPR @ 1% FPR ({cv.n_splits}-fold CV): {tprMean:.3f} +/- {tprStd:.3f}")

model = make_pipeline(
        SimpleImputer(missing_values=-1, strategy='mean'),
        GaussianNB()
    )

auc_scores = cross_val_score(
     model, X, y,
     cv=cv,
     scoring="roc_auc",
     n_jobs=-1
 )
 

tprMean, tprStd = tpr_at_fpr_cv(model, X, y, cv=cv)
 
print(f"Classifier: Gaussian NB, Mean AUC {cv.n_splits}-fold CV: {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}, Mean TPR @ 1% FPR ({cv.n_splits}-fold CV): {tprMean:.3f} +/- {tprStd:.3f}")

