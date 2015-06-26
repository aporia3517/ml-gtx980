import os, sys
import theano
import theano.tensor as T
from logistic_sgd import LogisticRegression, load_data
import numpy as np
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D

from sklearn import decomposition
from sklearn import datasets

np.random.seed(5)
dataset = 'mnist.pkl.gz'
mnist = load_data(dataset)
X, y = mnist[0]
X = X.get_value()
y = y.eval()

fig = pl.figure(1, figsize=(12,9))
pl.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
pl.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)
centers = [[1, 1], [-1, -1], [1, -1]]

for number in range(10):
    ax.text3D(X[y == number, 0].mean(),
              X[y == number, 1].mean() + 1.5,
              X[y == number, 2].mean(), str(number),
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'),
             )
# Reorder the labels to have colors matching the cluster results
#y = np.choose(y, [1, 2, 0]).astype(np.float)
#ax.scatter(X[:, 0], X[:, 1], X[:, 2], cmap=pl.cm.spectral)
p = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=pl.cm.spectral)
pl.colorbar(p)

x_surf = [X[:, 0].min(), X[:, 0].max(),
          X[:, 0].min(), X[:, 0].max()]
y_surf = [X[:, 0].max(), X[:, 0].max(),
          X[:, 0].min(), X[:, 0].min()]
x_surf = np.array(x_surf)
y_surf = np.array(y_surf)
v0 = pca.transform(pca.components_[0])
v0 /= v0[-1]
v1 = pca.transform(pca.components_[1])
v1 /= v1[-1]

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

pl.show()
