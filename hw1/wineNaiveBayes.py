from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# For readability
cols = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "ash_alc",      # alkalinity of ash
    "magnesium",
    "tot_phenols",
    "flav",
    "nonflav_phen",
    "proanth",
    "color_int",
    "hue",
    "od280_od315",
    "proline"
]

df = pd.read_csv("wine.data.csv", header= None, names=cols)

# Normalize features
df_z = df.copy()
df_z[df.columns[1:]] = df[df.columns[1:]].apply(
    lambda col: (col - col.mean()) / col.std()
)

# Split into features (tabular), classes (vector)
X, y = df_z.iloc[:, 1:], df_z.iloc[:,0]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

# -- GAUSSIAN NAIVE BAYES --

# Fit & Predict
gaussian_y_pred = GaussianNB().fit(X=X_train, y=y_train).predict(X_test)

# Evaluate
gaussian_acc = accuracy_score(y_test, gaussian_y_pred)

print(f"Gaussian Naive Bayes' Accuracy: {gaussian_acc:.4f}")

# -- BERNOULLI NAIVE BAYES --

# Some binarize values to try
binarize_arr = np.arange(0,3,0.025)

results = []

# Fit to multiple binarize values, append to list
for binarize in binarize_arr:
    bernoulli_y_pred = BernoulliNB(binarize=binarize).fit(X=X_train, y=y_train).predict(X_test)
    
    bernoulli_acc = accuracy_score(y_test, bernoulli_y_pred)
    
    results.append((binarize, bernoulli_acc))
    
# Print top 5 results
results.sort(reverse=True, key=lambda x: x[1])

for result in results[0:4]:
    print(f"Bernoulli Naive Bayes' Accuracy (Binarize={result[0]:.4f}): {result[1]:.4f}")