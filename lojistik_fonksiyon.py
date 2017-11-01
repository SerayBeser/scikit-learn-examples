# coding=utf-8
# Orjinali:  http://scikit-learn.org
""" Lojistik Fonksiyon """

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model

# Bu sadece duz bir dogru, test edebilmek icin
# Gaussian gurultusu iceriyor
xmin, xmax = -5, 5

# uzerinde calisacagimiz veri sayisi
sample_number = 100

# cekirdegi sifirliyoruz
np.random.seed(0)

# uzerinde calisacagimiz verileri hazirliyoruz
# sentetik veri setimiz
X = np.random.normal(size=sample_number)
y = (X > 0).astype(np.float)
X[X > 0] *= 4
X += .3 * np.random.normal(size=sample_number)
X = X[:, np.newaxis]

# Logistik Regresyon icin siniflandiricimiz
clf = linear_model.LogisticRegression(C=1e5)
clf.fit(X, y)

plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)


def model(x):
    return 1 / (1 + np.exp(-x))


loss = model(X_test * clf.coef_ + clf.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)
# Logistik Regresyon'u plot'a yerlestirdik.

# Lineer Regresyon icin siniflandiricimiz
ols = linear_model.LinearRegression()
ols.fit(X, y)
plt.plot(X_test, ols.coef_ * X_test + ols.intercept_, linewidth=1)
plt.axhline(.5, color='.5')

plt.ylabel('y')
plt.xlabel('X')
plt.xticks(range(-5, 10))
plt.yticks([0, 0.5, 1])
plt.ylim(-.25, 1.25)
plt.xlim(-4, 10)
plt.legend(('Logistic Regression Model', 'Linear Regression Model'),
           loc="lower right", fontsize='small')
plt.show()
