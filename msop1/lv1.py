import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, t
import scipy.stats as stats

data = pd.read_csv("studenti.csv")

print("Prvih 5 redaka skupa podataka:\n", data.head(), "\n")
print("Osnovne informacije o skupu podataka:\n")
print(data.info(), "\n")

print("Nedostajuće vrijednosti po stupcima:\n", data.isnull().sum(), "\n")

data = data.dropna()
data = data.drop_duplicates()

print("Broj redaka i stupaca nakon čišćenja:", data.shape, "\n")

print("Deskriptivna statistika:\n", data.describe(), "\n")

print("Jedinstvene vrijednosti za varijablu 'Spol':", data["Spol"].unique(), "\n")

corr_matrix = data.corr(numeric_only=True)
print("Korelacijska matrica:\n", corr_matrix, "\n")

plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelacijska matrica - toplinska karta")
plt.show()

plt.figure(figsize=(6,4))
plt.hist(data["Sati_ucenja_tjedno"], bins=10, edgecolor="black")
plt.title("Distribucija sati učenja tjedno")
plt.xlabel("Sati učenja")
plt.ylabel("Broj studenata")
plt.show()

plt.figure(figsize=(5,4))
sns.boxplot(x=data["Zavrsna_ocjena"])
plt.title("Boxplot - Završna ocjena")
plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x="Spol", data=data)
plt.title("Distribucija po spolu")
plt.show()

sns.pairplot(data, hue="Spol")
plt.suptitle("Pairplot - odnosi između varijabli", y=1.02)
plt.show()

print("Mjere središnje tendencije i oblika:\n")
print("Srednja vrijednost (Zavrsna_ocjena):", data["Zavrsna_ocjena"].mean())
print("Medijan:", data["Zavrsna_ocjena"].median())
print("Standardna devijacija:", data["Zavrsna_ocjena"].std())
print("Skewness:", skew(data["Zavrsna_ocjena"]))
print("Kurtosis:", kurtosis(data["Zavrsna_ocjena"]), "\n")