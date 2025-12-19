# coding:utf8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math
import os

# ===============================
# Fonction pour ouvrir les fichiers
def ouvrirUnFichier(nom):
    try:
        return pd.read_csv(nom, sep=",", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(nom, sep=",", encoding="latin1")
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier {nom} : {e}")
        return None

# chemins de base
BASE_PATH = r"C:\Python\seance_6\src\data"
CHEMIN_ILES = os.path.join(BASE_PATH, "island-index (1).csv")
CHEMIN_MONDE = os.path.join(BASE_PATH, "Le-Monde-HS-Etats-du-monde-2007-2025 (1).csv")

# ===============================
# PARTIE SUR LES ÎLES
print("partie sur les îles !")
iles = pd.DataFrame(ouvrirUnFichier(CHEMIN_ILES))
print("Colonnes du fichier îles :", iles.columns.tolist())

# conversion en données logarithmiques
def conversionLog(liste):
    log = []
    for element in liste:
        # fonction math.log est ln (logarithme naturel)
        if element > 0:
             log.append(math.log(element))
    return log

# tri par ordre décroissant des listes îles et pop
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

# Fonction pour obtenir le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    # pop et etat même longueur de liste
    ordrepop = []
    for element in range(0, len(pop)):
        # np.isnan(pop[element]), vérifier que la pop existe
        if not np.isnan(pop[element]):
            ordrepop.append([float(pop[element]), etat[element]])
            
    # Tri par ordre décroissant (basé sur la pop, le premier élément)
    ordrepop = ordreDecroissant(ordrepop)
    
    # Remplacement de la population par le rang
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
        
    return ordrepop

# Fonction pour obtenir l'ordre défini entre deux classements (listes spécifiques aux populations)
def classementPays(ordre1, ordre2):
    # Simplification : Les indices doivent être gérés dans les boucles for, pas en dur.
    classement = []
    
    # pour ne pas dépasser les bornes
    len1 = len(ordre1)
    len2 = len(ordre2)

    if len1 <= len2:
        # Boucle sur l'ordre le plus grand (ordre2) pour chercher l'élément dans l'ordre1
        for element2 in range(len2):
            for element1 in range(len1):
                # Si le nom de l'État correspond
                if ordre2[element2][1] == ordre1[element1][1]:
                    # [Rang1 (ordre1), Rang2 (ordre2), Nom de l'État]
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element1][1]])
                    break
    else: 
        for element1 in range(len1):
            for element2 in range(len2):
                if ordre2[element2][1] == ordre1[element1][1]:
                    classement.append([ordre1[element1][0], ordre2[element2][0], ordre1[element1][1]])
                    break
    
    return classement

# =======================================
# PARTIE SUR LES ÎLES (Loi Rang-Taille)
print("partie sur les îles : Loi Rang-Taille")
# QUESTION 2. Ouvrir le fichier
iles = pd.DataFrame(ouvrirUnFichier(CHEMIN_ILES))

# QUESTION 3. Isoler la colonne « Surface (km²) » et ajouter les continents
# Attention! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().
print("QUESTION 3 : isoler colonne des surfaces Km² et ajout des continents")
# Normalisation des noms de colonnes
iles.columns = iles.columns.str.strip()
iles.columns = iles.columns.str.lower() #pour minuscules
print("Colonnes normalisées :", iles.columns.tolist())

# Recherche automatique de la bonne colonne
possible_names = [c for c in iles.columns if "surface" in c]
if len(possible_names) == 0:
    raise ValueError("Impossible de trouver une colonne contenant 'surface'")
col_surface = possible_names[0]  # prend la 1e
surfaces_iles = iles[col_surface].tolist()
surfaces_iles = [float(s) for s in surfaces_iles if pd.notna(s)]

surfaces_iles = [float(s) for s in surfaces_iles if pd.notna(s)] # Nettoyage des NaN et cast

# Ajout des continents (en km²)
surfaces_iles.extend([
    85545323.0, # Asie/Afrique / Europe
    37856841.0, # Amérique
    7768030.0,  # Antarctique
    7605049.0   # Australie
])

# Création d'une liste des noms/labels correspondants pour référence (non demandé mais utile)
noms_iles = iles['toponyme'].tolist()
noms_iles.extend(['Asie/Afrique/Europe', 'Amérique', 'Antarctique', 'Australie'])

# QUESTION 4. Ordonner la liste obtenue avec la fonction locale ordreDecroissant()
print("QUESTION 4")
surfaces_triees = ordreDecroissant(surfaces_iles)

# QUESTION 5. Visualiser la loi rang-taille en créant une image de sortie.
# Le rang est simplement l'indice + 1.
print("QUESTION 5")
rangs = np.arange(1, len(surfaces_triees) + 1)

plt.figure(figsize=(10, 6))
plt.plot(rangs, surfaces_triees, marker='o', linestyle='-', color='blue')
plt.title("Loi Rang-Taille des Surfaces (Échelle Linéaire)")
plt.xlabel("Rang")
plt.ylabel("Surface (km²)")
plt.grid(True)
plt.show()

# QUESTION 6. L'image obtenue est illisible. Il vous faut convertir les axes en logarithme.
# Utiliser la fonction locale conversionLog() proposée : 
#APPLICATION
print("question 6 : fonction locale conversionLog() proposée")
surfaces_log = conversionLog(surfaces_triees)
rangs_log = conversionLog(rangs.tolist()) # Convertir les rangs en log

plt.figure(figsize=(10, 6))
# Option 1: Afficher les données log(taille) vs log(rang)
print("option 1")
plt.plot(rangs_log, surfaces_log, marker='o', linestyle='-', color='red')
plt.title("Loi Rang-Taille des Surfaces (Échelle Log-Log)")
plt.xlabel("Log(Rang)")
plt.ylabel("Log(Surface)")
plt.grid(True)

# Option 2 (Alternative visuelle): utilisation de l'échelle log sur Matplotlib direct
print("option 2")
plt.figure(figsize=(10, 6))
plt.loglog(rangs, surfaces_triees, marker='o', linestyle='-', color='green')
plt.title("Loi Rang-Taille des Surfaces (Échelle Log-Log via plt.loglog)")
plt.xlabel("Rang")
plt.ylabel("Surface (km²)")
plt.grid(True)
plt.show()

# QUESTION 7. Est-il possible de faire un test sur les rangs? (mettre votre réponse sous la forme d’un commentaire dans le fichier)
print("\nQUESTION 7 REPONSE ")
print(" Non, il n'est pas possible de faire un test statistique paramétrique (comme le test de Student ou Shapiro-Wilks) directement sur les rangs.")
print("Les rangs ne sont pas des données continues issues d'une distribution de probabilité mais des données ordinales artificiellement créées.")
print("cependant, une fois transformée en échelle loglog, la relation linéaire (loi de Zipf)peut être testée par régression linéaire (test de corrélation, p-value du coefficient de régression).")


#Attention ! Il va falloir utiliser des fonctions natives de Python dans les fonctions locales que je vous propose pour faire l'exercice. Vous devez caster l'objet Pandas en list().

# =======================================================
# PARTIE SUR LES POPULATIONS DES ÉTATS DU MONDE
print("Partie sur les Populations des états du monde")

#Source. Depuis 2007, tous les ans jusque 2025, M. Forriez a relevé l'intégralité du nombre d'habitants dans chaque États du monde proposé par un numéro hors-série du monde intitulé États du monde. Vous avez l'évolution de la population et de la densité par année.
monde = pd.DataFrame(ouvrirUnFichier(r"C:\Python\seance_6\src\data\Le-Monde-HS-Etats-du-monde-2007-2025 (1).csv"))

# QUESTION 8 : création et QUESTION 9 : ouverture du fichier
monde = pd.DataFrame(ouvrirUnFichier(r"C:\Python\seance_6\src\data\Le-Monde-HS-Etats-du-monde-2007-2025 (1).csv"))

# QUESTION 10 Isoler les colonnes « État », « Pop 2007 », « Pop 2025 », « Densité 2007 » et « Densité 2025 »
print("\nQUESTION 10 : isoler les colonnes des états")
if not monde.empty:
    # Extraction des colonnes
    etats_monde = monde['État'].tolist()
    pop_2007 = monde['Pop 2007'].tolist()
    pop_2025 = monde['Pop 2025'].tolist()
    densite_2007 = monde['Densité 2007'].tolist()
    densite_2025 = monde['Densité 2025'].tolist()

#préparation des fonctions locales    
    print("\nPartie Populations Mondiales")
    print(f"Extraction réussie de {len(etats_monde)} états et des données de population/densité.")
    print("listes prêtes pour les fonctions d'ordre et de classement")
else:
    print("\nErreur vérifier chemin?")

#=================
#QUESTION 11
print("\nQUESTION 11 : Classements par population et densité")

# Classement des populations
ordre_pop_2007 = ordrePopulation(pop_2007, etats_monde)
ordre_pop_2025 = ordrePopulation(pop_2025, etats_monde)

# Classement des densités
ordre_dens_2007 = ordrePopulation(densite_2007, etats_monde)
ordre_dens_2025 = ordrePopulation(densite_2025, etats_monde)

print("Classement Pop 2007 :")
print(ordre_pop_2007[:10])   # affichage des 10 premiers pour vérifier

print("\nClassement Pop 2025 :")
print(ordre_pop_2025[:10])

print("\nClassement Densité 2007 :")
print(ordre_dens_2007[:10])

print("\nClassement Densité 2025 :")
print(ordre_dens_2025[:10])

#=================
# QUESTION 12
print("QUESTION 12 : utilisation de classementPays et Tri")
classement_compare = classementPays(ordre_pop_2007, ordre_dens_2007)
classement_compare.sort(key=lambda x: x[0])

print("\nQUESTION 12: Comparaison Population vs. Densité (2007) ===")
    # Affichage du résultat final (les 10 premiers pays classés par Population 2007)
print("| Rang Pop 2007 | Rang Densité 2007 | État |")
print("| :---: | :---: | :--- |")
for rang_pop, rang_densite, etat in classement_compare[:10]:
    print(f"| {rang_pop} | {rang_densite} | {etat} |")
    
    # Stocker le résultat
    classement_2007_final = classement_compare
    


#============================
#QUESTION 13 : 
print("QUESTION 13 : isoler les 2 colonnes")

if 'classement_2007_final' in locals() and classement_2007_final: 
    
    rangs_pop = []     # Rang Population 2007 
    rangs_densite = [] # Rang Densité 2007
    
    for rang_pop, rang_densite, _ in classement_2007_final:
        rangs_pop.append(rang_pop)
        rangs_densite.append(rang_densite)

    print("\nQUESTION 13: Extraction des Rangées Numériques réussie")
    # print(f"Premiers rangs Pop: {rangs_pop[:5]}")
    # print(f"Premiers rangs Densité: {rangs_densite[:5]}")
    
    # S'assurer que les listes sont des tableaux NumPy (souvent requis par scipy)
    rangs_pop_np = np.array(rangs_pop)
    rangs_densite_np = np.array(rangs_densite)
    
    #==============================
    # QUESTION 14 
    print("QUESTION 14 calcul des coefficients de corrélation des rangs")
    
    # 1. Coefficient de Corrélation de Spearman (rho)
    # Mesure la force et la direction de la relation monotone entre les deux jeux de rangs.
    coef_spearman, p_spearman = scipy.stats.spearmanr(rangs_pop_np, rangs_densite_np)
    
    # 2. Coefficient de Corrélation de Kendall (tau)
    # Mesure la concordance des deux jeux de rangs (probabilité que les rangs relatifs correspondent).
    coef_kendall, p_kendall = scipy.stats.kendalltau(rangs_pop_np, rangs_densite_np)
    
    print("\nQUESTION 14 : Coefficients de Corrélation des Rangs")
    
    print("Corrélation entre le classement par Population 2007 et Densité 2007 :")
    print(f"1. Spearman (rho) : {coef_spearman:.4f} (p-value: {p_spearman:.4f})")
    print(f"2. Kendall (tau) : {coef_kendall:.4f} (p-value: {p_kendall:.4f})")
    
    # Interprétation pour le Rapport
    
    print("\nAnalyse pour le Rapport")
    if p_spearman < 0.05:
        print("La corrélation est statistiquement significative (p < 0.05).")
    else:
        print("La corrélation n'est PAS statistiquement significative (p > 0.05).")
        
    print(f"Le coefficient de Spearman ({coef_spearman:.4f}) est proche de 0, indiquant une relation monotone (tendance) très FAIBLE entre le classement par Population et par Densité.")
    print("Conclusion : La taille d'un État (population) n'est pas un bon prédicteur de sa densité de population, et inversement.")

else:
    print("\nerreur")
