# coding=utf-8
# Orjinali:  http://scikit-learn.org
""" Scikit-Learn: Boston Ev Fiyatlari Tahmini """

from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

# Model Secimi
# Tahminci Olarak Lineer Regresyon Secildi
linear_regression_estimator = linear_model.LinearRegression()

# Veri Setimizin Yuklenmesi
# Veri Setinde 506 Adet Eve Ait Bilgiler Bulunuyor.
boston_houses = datasets.load_boston()

# Denetimli Ogrenme icin
# Sonucunu Tahmin Edilmeye Calisilacak Hedef Degisken
y = boston_houses.target

# 5-Katli Capraz Dogrulama Kullanilacak
# Tahmin Sonuclari
predicted = cross_val_predict(linear_regression_estimator, boston_houses.data, y, cv=5)

# Gorsellestirme
fig, ax = plt.subplots()
ax.scatter(y, predicted, marker="o", s=5)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k', lw=2)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
