# coding=utf-8
# Orjinali:  http://scikit-learn.org
""" Scikit-Learn: Polinomsal Regresyon ile Asiri ve Yetersiz Uyum Örnegi"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Tohum sifirlanmasiyla ayni sayi seti her seferinde gorunecektir.
# Rastgele tohum sifirlanmazsa, her cagirrmada farkli sayilar gorunur.

# Tohumu sifirliyoruz cunku hep ayni sayi seti uzerinde calismak istiyoruz.
np.random.seed(0)

# kac adet ornek uzerinde calisagimizi ayarliyoruz.
sample_number = 30

# hedef orneklerimize uyabilecek polinomu saglamak icin
# polinom derecelerini belirliyoruz.
degrees = [1, 4, 15]

# fonksiyonumuzu hesaplayalim
function = lambda X: np.cos(1.5 * np.pi * X)
X = np.sort(np.random.rand(sample_number))
y = function(X) + np.random.randn(sample_number) * 0.1

# plot
plt.figure(figsize=(14, 5))

# her polinom derecesi icin calistiralim
for i in range(len(degrees)):
    # plot
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()

    # boru hatti
    # Boru hattinin amaci,
    # farkli parametreler ayarlanirken
    # birbiriyle capraz gecerliligi olan birkac adimi bir araya getirmektir.
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Modeli capraz dogrulama ile hesaplama
    scores = cross_val_score(pipeline, X[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    X_test = np.linspace(0, 1, 100)  # plot
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, function(X_test), label="True function")
    plt.scatter(X, y, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0, 1))
    plt.ylim((-2, 2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()
