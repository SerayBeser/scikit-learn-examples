# coding=utf-8
# Orjinali:  http://scikit-learn.org
""" Scikit-Learn: Lojistik Regresyon ve iris Veri Seti """

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets

# Veri setimizi yukluyoruz.
iris = datasets.load_iris()

# Biz sadece ilk iki ozelligi kullanacagiz.
X = iris.data[:, :2]
Y = iris.target

# orgudeki adim sayisi
step_size = .02

# lojistik regresyon
logistic_regression = linear_model.LogisticRegression(C=1e5)

# Neighbours Classifier ornegini olusturduk ve verileri uygun hale getirdik.
logistic_regression.fit(X, Y)

# Karar sinirini cizmek icin. Bunun icin her birine bir renk atiyoruz.

# Mesh noktasini isaretlemek icin [x_min, x_max] x [y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# Modelimiz tahminini gerceklestiriyor.
Z = logistic_regression.predict(np.c_[xx.ravel(), yy.ravel()])

# Sonuclari plot'a ekliyoruz.
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Egitim verilerimizide plot'a ekliyoruz.
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
