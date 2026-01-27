import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Make two blobs
X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.0, random_state=42)

clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X, y)

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

