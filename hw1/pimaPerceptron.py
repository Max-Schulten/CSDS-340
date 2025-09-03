from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from perceptron_logic import Perceptron
from pprint import pprint

cols = [
    "preg",
    "glu",
    "bp",
    "skin",
    "insu",
    "bmi",
    "dpf",
    "age",
    "class"
]

df = pd.read_csv("pima-indians-diabetes.csv", header=None, names=cols)

X, y = df.iloc[:,:-1].to_numpy(dtype=float), df.iloc[:,-1].to_numpy(dtype=float)
X_mean = X.mean(axis=0)
X_std  = X.std(axis=0, ddof=0)
X_z = (X - X_mean) / X_std

results = []
best_perceptron = None

for eta in np.logspace(0, -6, 7):
    perceptron = Perceptron(eta=eta, random_state=1, n_iter=50)
    fitted_perceptron = perceptron.fit(X_z, y)
    results.append((eta, fitted_perceptron.errors_[-1]))
    
    if not best_perceptron:
        best_perceptron = perceptron
    elif best_perceptron.errors_[-1] > fitted_perceptron.errors_[-1]:
        best_perceptron = fitted_perceptron

results.sort(key=lambda x: x[1])

pprint(results)
print(f"BEST ACCURACY ACHIEVE FOR ETA={float(results[0][0])}: {1-results[0][1]/X_z.shape[0]}")

plt.plot(range(1, len(best_perceptron.errors_) + 1), best_perceptron.errors_, marker='o') # type: ignore
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.suptitle('Perceptron Training on Pima Indians Dataset')
plt.title(f'eta = {best_perceptron.eta}') # type: ignore
plt.show()