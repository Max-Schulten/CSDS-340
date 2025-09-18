import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

X = np.array(
    [
     [0, 1],
     [0, -1],
     [1, 0],
     [-1, 0],
     [0, 2],
     [0, -2],
     [2, 0],
     [-2,0]
     ]
    )

y = [1, 1, 1, 1, 2, 2, 2, 2]


classifier = SVC(C = 1e6) # Extremely large C for hard margin
classifier.fit(X, y)

# TAKEN IN PART FROM https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python
def make_meshgrid(x, y):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02), np.arange(y_min, y_max, .02))
    return xx, yy

def plot_contours(ax, clf, xx, yy):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, cmap=plt.cm.berlin, alpha=0.8)
    return out

fig, ax = plt.subplots()
# title for the plots
title = ('Decision surface of RBF Kernel SVC')
X1, X2 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X1, X2)

plot_contours(ax, classifier, xx, yy)
ax.scatter(X1, X2, c=y, cmap=plt.cm.berlin, s=20, edgecolors='k')
ax.set_ylabel('x_1')
ax.set_xlabel('x_2')
ax.set_xticks(())
ax.set_yticks(())
ax.set_title(title)
ax.legend()
plt.show()