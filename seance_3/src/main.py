# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/

# Sources des données : production de M. Forriez, 2016-2023

import pandas as pd
import numpy as np

# ===============================
# csv
chemin_csv = r"C:\Python\seance_3\src\data\resultats-elections-presidentielles-2022-1er-tour.csv"
df = pd.read_csv(chemin_csv)

print("Colonnes détectées :", df.columns.tolist())

# =================================
# colonnes quantitatives (variables numériques)

colonnes_quant = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nColonnes quantitatives :", colonnes_quant)

# =============================
# QUESTION 5 - calcul des paramètres : moyenne, médiane, mode, etc...

stats = []

for col in colonnes_quant:
    serie = df[col]

    moyenne = serie.mean()
    mediane = serie.median()
    mode = serie.mode().iloc[0] if not serie.mode().empty else None
    ecart_type = serie.std()
    ecart_absolu = np.abs(serie - moyenne).mean()
    etendue = serie.max() - serie.min()

    stats.append([
        col,
        moyenne,
        mediane,
        mode,
        ecart_type,
        ecart_absolu,
        etendue
    ])

# conversion en DataFrame + arrondi à 2 décimales
df_stats = pd.DataFrame(
    stats,
    columns=[
        "Colonne", "Moyenne", "Médiane", "Mode",
        "Écart-type", "Écart absolu à la moyenne", "Étendue"
    ]
).round(2)

print("\nSTATISTIQUES")
print(df_stats)

# ===============================
# QUESTION 7 - Distance interquartile et interdécile

df_stats["IQR (Q3-Q1)"] = [
    df[col].quantile(0.75) - df[col].quantile(0.25)
    for col in colonnes_quant
]

df_stats["Interdécile (D9-D1)"] = [
    df[col].quantile(0.90) - df[col].quantile(0.10)
    for col in colonnes_quant
]

df_stats = df_stats.round(2)

print("\nQUESTION 7 - STATISTIQUES AVEC IQR ET INTERDÉCILE")
print(df_stats)

# =========================================
# QUESTION 8 - Boîtes à moustaches

dossier_box = r"C:\Python\seance_3\src\images\boxplots"
os.makedirs(dossier_box, exist_ok=True)

for col in colonnes_quant:
    plt.figure(figsize=(6, 8))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boîte à moustache – {col}")
    plt.ylabel(col)

    plt.savefig(os.path.join(dossier_box, f"box_{col}.png"), dpi=150)
    plt.close()

print("\nQUESTION 8 - Boxplots enregistrés dans : ", dossier_box)

# =====================================
# Question 10
print("QUESTION 10")

chemin_islands = r"C:\Python\seance_3\src\data\island-index (1).csv"
df_islands = pd.read_csv(chemin_islands)

print("\nColonnes îles :", df_islands.columns.tolist()) 

surface = df_islands["Surface (km²)"]


# dictionnaire de comptage
compte = {
    "0-10": 0,
    "10-25": 0,
    "25-50": 0,
    "50-100": 0,
    "100-2500": 0,
    "2500-5000": 0,
    "5000-10000": 0,
    "10000+": 0
}

for s in surface:
    if 0 < s <= 10:
        compte["0-10"] += 1
    elif 10 < s <= 25:
        compte["10-25"] += 1
    elif 25 < s <= 50:
        compte["25-50"] += 1
    elif 50 < s <= 100:
        compte["50-100"] += 1
    elif 100 < s <= 2500:
        compte["100-2500"] += 1
    elif 2500 < s <= 5000:
        compte["2500-5000"] += 1
    elif 5000 < s <= 10000:
        compte["5000-10000"] += 1
    elif s > 10000:
        compte["10000+"] += 1

print("\nCATÉGORIES DES ÎLES")
for k, v in compte.items():
    print(f"{k} km² : {v} îles")

