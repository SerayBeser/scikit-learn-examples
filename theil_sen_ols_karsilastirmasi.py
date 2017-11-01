# coding=utf-8
# Orjinali:  http://scikit-learn.org
""" Scikit-Learn: Theil-Sen ve Siradan En Kucuk Kareler Tahminci Karsilastirmasi """

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, TheilSenRegressor

# Tahmincilerimizi ayarliyoruz.
estimators = [('Ordinary Least Squares', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42))]

# Plot'da yer alacak dogrulara ait bilgileri tanimliyoruz.
colors = {'Ordinary Least Squares': 'turquoise', 'Theil-Sen': 'gold', }
# cizgi kalinligi (line width)
lw = 2

##############################################################################
# Outliers only in the y direction

# Tohum sifirlanmasiyla ayni sayi seti her seferinde gorunecektir.
# Rastgele tohum sifirlanmazsa, her cagirrmada farkli sayilar gorunur.
# Tohumu sifirliyoruz cunku hep ayni sayi seti uzerinde calismak istiyoruz.
np.random.seed(0)

# kac adet ornek uzerinde calisagimizi ayarliyoruz.
sample_number = 200

# Modelimizi Ayarliyoruz.
# Lineer model y = 3*x + N(2, 0.1**2)
x = np.random.randn(sample_number)
w = 3.
c = 2.,
# gurultu ekliyoruz.
noise = 0.1 * np.random.randn(sample_number)
y = w * x + c + noise
# 10% aykiri veri
y[-20:] += -20 * x[-20:]
X = x[:, np.newaxis]

# Orneklerimizi plot'a yerlestiriyoruz.
plt.scatter(x, y, color='indigo', marker='x', s=40)

# Tahmin Edecegimiz Dogru.
line_x = np.array([-3, 3])

# Her Tahminci Icin
for name, estimator in estimators:
    # ne kadar surede uyum(fit) ettigini gormek icin giris zamani.
    t0 = time.time()

    # Tahminci, uyum(fit) gerceklestiriyor.
    estimator.fit(X, y)

    # uyum(fit) zamani
    elapsed_time = time.time() - t0

    # Tahminci, tahminini gerceklestiriyor.
    y_pred = estimator.predict(line_x.reshape(2, 1))

    # Tahmincileri Plot'a Ekliyoruz.
    plt.plot(line_x, y_pred, color=colors[name], linewidth=lw,
             label='%s (fit time: %.2fs)' % (name, elapsed_time))

    # Plot Bilgilerini Ayarliyoruz.
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.title("Corrupt y")

##############################################################################
# Outliers only in the x direction

np.random.seed(0)
# Linear model y = 3*x + N(2, 0.1**2)
x = np.random.randn(sample_number)
noise = 0.1 * np.random.randn(sample_number)
y = 3 * x + 2 + noise
# 10% outliers
x[-20:] = 9.9
y[-20:] += 22
X = x[:, np.newaxis]

plt.figure()
plt.scatter(x, y, color='indigo', marker='x', s=40)

line_x = np.array([-3, 10])
for name, estimator in estimators:
    t0 = time.time()
    estimator.fit(X, y)
    elapsed_time = time.time() - t0
    y_pred = estimator.predict(line_x.reshape(2, 1))
    plt.plot(line_x, y_pred, color=colors[name], linewidth=lw,
             label='%s (fit time: %.2fs)' % (name, elapsed_time))

    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.title("Corrupt x")
plt.show()
