# coding=utf-8
""" Scikit-Learn: K-Ortalama ve Iris Veri Seti """

from sklearn.cluster import KMeans
from sklearn import datasets

import matplotlib.pyplot as plt

# iris veri setini yukleyelim
iris = datasets.load_iris()
X, y = iris.data, iris.target
print X[:, 1]

# 3 ayri merkez ile kumele islemi
k_means = KMeans(n_clusters=3, random_state=0)
k_means.fit(X)
y_pred = k_means.predict(X)

# dagilim grafigi
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='Dark2')
plt.show()
