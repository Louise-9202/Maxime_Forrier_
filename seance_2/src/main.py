# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ================================
# fichier csv

chemin_csv = r"C:\Python\seance_3\src\data\resultats-elections-presidentielles-2022-1er-tour.csv"

df = pd.read_csv(chemin_csv)
print("Colonnes détectées :", df.columns.tolist())

# sélection colonne "Libellé du département"
col_dept = [c for c in df.columns if "Libell" in c][0]
print("Colonne département trouvée :", col_dept)

# ====================================
# Nettoyage des noms pour les fichiers

def clean_name(name: str) -> str:
    """Nettoie un texte pour l'utiliser comme nom de fichier."""
    if not isinstance(name, str):
        name = str(name)
    forbidden = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for f in forbidden:
        name = name.replace(f, '_')
    return name


# ============================================
# dossiers de sortie

dossier = r"C:\Python\seance_3\src\images"
os.makedirs(dossier, exist_ok=True)

dossier_BNEA = os.path.join(dossier, "pie_BNEA")
os.makedirs(dossier_BNEA, exist_ok=True)

dossier_voix = os.path.join(dossier, "voix_candidats")
os.makedirs(dossier_voix, exist_ok=True)


# =======================================================
# histogramme des inscrits

plt.figure(figsize=(10, 6))
plt.hist(df["Inscrits"], bins=20, density=True, edgecolor="black")
plt.title("Histogramme de la distribution des inscrits")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Densité")

plt.savefig(os.path.join(dossier, "histogramme_inscrits.png"), dpi=150)
plt.close()
print("Histogramme créé")

# =======================================================
# QUESTION 12 diagrammes circulaires : blans/ nuls/ exprimés/ abstentions
print("QUESTION 12 diagrammes circulaires")
col_blancs = "Blancs"
col_nuls = "Nuls"
col_expr = [c for c in df.columns if "Exprim" in c][0]
col_abst = "Abstentions"

print("Colonnes BNEA :", col_blancs, col_nuls, col_expr, col_abst)

for i in range(len(df)):
    dept = df.loc[i, col_dept]
    dept_clean = clean_name(dept)

    valeurs = [
        df.loc[i, col_blancs],
        df.loc[i, col_nuls],
        df.loc[i, col_expr],
        df.loc[i, col_abst]
    ]
    labels = ["Blancs", "Nuls", "Exprimés", "Abstentions"]

    plt.figure(figsize=(8, 8))
    plt.pie(valeurs, labels=labels, autopct="%1.1f%%")
    plt.title(f"BNEA – {dept}")

    plt.savefig(os.path.join(dossier_BNEA, f"BNEA_{dept_clean}.png"), dpi=150)
    plt.close()

print("Diagrammes circulaires BNEA créés !")
# =======================================================
# diagrammes voix par candidats

# Détection auto des colonnes
colonnes_voix = [col for col in df.columns if "Voix" in col]
colonnes_nom = [col for col in df.columns if "Nom" in col and "Liste" not in col]
colonnes_prenom = [col for col in df.columns if "Prénom" in col or "Prenom" in col]

print("Colonnes des voix détectées :", colonnes_voix)
print("Colonnes des noms :", colonnes_nom)
print("Colonnes des prénoms :", colonnes_prenom)

# pour que les longueurs correspondent
nb_candidats = min(len(colonnes_voix), len(colonnes_nom), len(colonnes_prenom))

# ----------------------
# par départements
for i in range(len(df)):
    dept = df.loc[i, col_dept]
    dept_clean = clean_name(dept)

    valeurs = [df.loc[i, colonnes_voix[j]] for j in range(nb_candidats)]
    labels = [
        f"{df.loc[i, colonnes_nom[j]]} {df.loc[i, colonnes_prenom[j]]}"
        for j in range(nb_candidats)
    ]

    # enlève candidats 0 voix
    valeurs_f = []
    labels_f = []
    for v, l in zip(valeurs, labels):
        if v > 0:
            valeurs_f.append(v)
            labels_f.append(l)

    plt.figure(figsize=(9, 9))
    plt.pie(valeurs_f, labels=labels_f, autopct="%1.1f%%")
    plt.title(f"Voix par candidat – {dept}")

    plt.savefig(os.path.join(dossier_voix, f"voix_{dept_clean}.png"), dpi=150)
    plt.close()

print("Diagrammes candidats par département : OK !")


# ----------------------
# France entière
voix_fr = [df[colonnes_voix[j]].sum() for j in range(nb_candidats)]
labels_fr = [
    f"{df[colonnes_nom[j]].iloc[0]} {df[colonnes_prenom[j]].iloc[0]}"
    for j in range(nb_candidats)
]

# filtrer candidats à 0 voix
voix_F = []
labels_F = []
for v, l in zip(voix_fr, labels_fr):
    if v > 0:
        voix_F.append(v)
        labels_F.append(l)

plt.figure(figsize=(10, 10))
plt.pie(voix_F, labels=labels_F, autopct="%1.1f%%")
plt.title("Répartition des voix — France entière")

plt.savefig(os.path.join(dossier_voix, "voix_France.png"), dpi=150)
plt.close()

print("Diagramme circulaire voix France : OK !")

# =======================================================
# Calculs des statistiques

# Colonnes quantitatives = colonnes numériques
colonnes_quant = df.select_dtypes(include=[np.number]).columns
print("\nColonnes quantitatives :", list(colonnes_quant))

stats = {}

for col in colonnes_quant:
    serie = df[col]

    stats[col] = {
        "moyenne": serie.mean(),
        "médiane": serie.median(),
        "mode": serie.mode().iloc[0] if not serie.mode().empty else None,
        "écart_type": serie.std(),
        "écart_absolu_moyenne": np.abs(serie - serie.mean()).mean(),
        "étendue": serie.max() - serie.min(),
        "IQR (Q3 - Q1)": serie.quantile(0.75) - serie.quantile(0.25),  # distance interquartile
        "interdécile (D9 - D1)": serie.quantile(0.90) - serie.quantile(0.10)  # distance interdécile
    }

    
# Arrondir les statistiques à 2 décimales
for col in stats:
    stats[col] = {k: round(v, 2) for k, v in stats[col].items()}

print("\n=== Statistiques calculées ===")
for col, valeurs in stats.items():
    print(f"\n▶ {col}")
    for nom, v in valeurs.items():
        print(f"   - {nom} : {v}")

# =======================================================
# Boîtes à moustaches pour chaque colonne quantitative

dossier_box = os.path.join(dossier, "img")
os.makedirs(dossier_box, exist_ok=True)

for col in colonnes_quant:
    plt.figure(figsize=(6, 6))
    plt.boxplot(df[col].dropna())
    plt.title(f"Boîte à moustaches — {col}")
    plt.ylabel(col)

    nom_fichier = os.path.join(dossier_box, f"boxplot_{clean_name(col)}.png")
    plt.savefig(nom_fichier, dpi=150, bbox_inches="tight")
    plt.close()

print("Boxplots/Boîtes à moustaches enregistrés dans le dossier img/")

# =======================================================
# chemin csv

chemin = r"C:\Python\seance_3\src\data\island-index (1).csv"
print("\nERREUR: Le fichier n'a pas été trouvé à C:\\Python\\seance_3\\src\\data\\island-index (1).csv.")
df = pd.read_csv(chemin_csv)

# Charger DataFrame des îles
try:
    df_iles = pd.read_csv(r"C:\python\seance_3\src\data\island-index (1).csv", encoding='utf-8')
    print("\nFichier 'island-index.csv' chargé avec succès.")
    print("Colonnes des îles :", df_iles.columns.tolist())
except FileNotFoundError:
    print(f"\nERREUR: Le fichier n'a pas été trouvé à {"C:\Python\seance_3\src\data\island-index (1).csv"}.")
    df_iles = pd.DataFrame() # DataFrame vide pour éviter erreurs

# =======================================================
# Catégorisation des surfaces
if not df_iles.empty and "Surface (km²)" in df_iles.columns:
    col_surface = "Surface (km²)"
    
    # 1. limites (bins) et étiquettes (labels)
    # np.inf représente l'infini pour la dernière catégorie
    bins = [0, 10, 25, 50, 100, 2500, 5000, 10000, np.inf]
    
    # étiquettes de chaque catégorie (intervalle)
    labels = [
        "[0, 10]", 
        "]10, 25]", 
        "]25, 50]", 
        "]50, 100]", 
        "]100, 2500]", 
        "]2500, 5000]", 
        "]5000, 10000]", 
        "]10000, +inf["
    ]
    
    # surface traitée en tant que variable quantitative (float/int)
    surfaces = df_iles[col_surface].astype(float)
    
    # Application de la fonction pd.cut, `right=True` est la valeur par défaut: (limite_inférieure, limite_supérieure], utilisation de `include_lowest=True` pour inclure la valeur 0 dans le premier bin [0, 10].
    df_iles["Catégorie_Surface"] = pd.cut(
        surfaces, 
        bins=bins, 
        labels=labels, 
        right=True, 
        include_lowest=True
    )
    
    # Dénombrer le nombre d'îles par catégorie
    comptage_categories = df_iles["Catégorie_Surface"].value_counts().sort_index()
    
    print("\n=== Dénombrement des Îles par Catégorie de Surface ===")
    print(comptage_categories)

else:
    print("\nCatégorisation non exécutée: 'df_iles' est vide ou la colonne 'Surface (km²)' est manquante.")

# Bonus : Exportation des résultats
print("BONUS")

if 'comptage_categories' in locals():
    # Conversion en DataFrame pour meilleure exportation
    df_comptage = comptage_categories.reset_index()

    # titrer les colonnes
    df_comptage.columns = ["Catégorie de Surface (km²)", "Nombre d'îles"]
    
    # dossier de sortie
    dossier_sortie_bonus = r"C:\Python\seance_3\src\output_bonus"
    os.makedirs(dossier_sortie_bonus, exist_ok=True)
    
    # 1. format CSV à exporter
    chemin_csv_bonus = os.path.join(dossier_sortie_bonus, "comptage_categories_iles.csv")
    df_comptage.to_csv(chemin_csv_bonus, index=False, encoding='utf-8')
    print(f"\nRésultats exportés en CSV à : {chemin_csv_bonus}")
    
    # 2. format Excel
    chemin_excel_bonus = os.path.join(dossier_sortie_bonus, "comptage_categories_iles.xlsx")
    df_comptage.to_excel(chemin_excel_bonus, index=False, sheet_name='Comptage Iles')
    print(f"Résultats exportés en Excel à : {chemin_excel_bonus}")

