#coding:utf8

import pandas as pd
import math
import scipy
import scipy.stats
import os

#C'est la partie la plus importante dans l'analyse de données. 
# D'une part, elle n'est pas simple à comprendre tant mathématiquement que pratiquement. D'autre, elle constitue une application des probabilités. L'idée consiste à comparer une distribution de probabilité (théorique) avec des observations concrètes.  De fait, il faut bien connaître les distributions vues dans la séance précédente afin de bien pratiquer cette comparaison. Les probabilités permettent de définir une probabilité critique à partir de laquelle les résultats ne sont pas conformes à la théorie probabiliste. Il n'est pas facile de proposer des analyses de données uniquement dans un cadre univarié. 
# Vous utiliserez la statistique inférentielle principalement dans le cadre d'analyses multivariées.  La statistique univariée est une statistique descriptive.  Bien que les tests y soient possibles, comprendre leur intérêt et leur puissance d'analyse dans un tel cadre peut être déroutant. Peu importe dans quelle théorie vous êtes, l'idée de la statistique inférentielle est de vérifier si ce que vous avez trouvé par une méthode de calcul est intelligent ou stupide.  Est-ce que l'on peut valider le résultat obtenu ou est-ce que l'incertitude qu'il présente ne permet pas de conclure ? Peu importe également l'outil, à chaque mesure statistique, on vous proposera un test pour vous aider à prendre une décision sur vos résultats. Il faut juste être capable de le lire.

#Par convention, on place les fonctions locales au début du code après les bibliothèques.
def ouvrirUnFichier(chemin):
    """Ouvre le fichier CSV et retourne le DataFrame."""
    (r"C:\Python\seance_5\src\data\Echantillonnage-100-Echantillons.csv", "r")
    try:
        # Lire le fichier, sans utiliser la première ligne comme header
        df = pd.read_csv(chemin, sep=',', header=None) 
        
        # Renommer les colonnes selon les catégories de l'exercice
        df.columns = ['Pour', 'Contre', 'Sans opinion'] 
        
        # S'assurer que les données sont des nombres (pour le cas où l'en-tête était juste une ligne)
        # On va tenter de convertir toutes les colonnes en numérique (erreurs mises à NaN)
        df = df.apply(pd.to_numeric, errors='coerce')
        
        print(f"Fichier {os.path.basename(chemin)} chargé avec {len(df)} échantillons.")
        return df
    except FileNotFoundError:
        print(f"ERREUR : Fichier non trouvé à {chemin}")
        return pd.DataFrame()

#Théorie de l'échantillonnage (intervalles de fluctuation)
#L'échantillonnage se base sur la répétitivité.
print("Résultat sur le calcul d'un intervalle de fluctuation")

donnees = pd.DataFrame(ouvrirUnFichier(r"C:\Python\seance_5\src\data\Echantillonnage-100-Echantillons.csv"))

#Théorie de l'estimation (intervalles de confiance)
#L'estimation se base sur l'effectif.
print("Résultat sur le calcul d'un intervalle de confiance")

#Théorie de la décision (tests d'hypothèse)
#La décision se base sur la notion de risques alpha et bêta.
#Comme à la séance précédente, l'ensemble des tests se trouve au lien : https://docs.scipy.org/doc/scipy/reference/stats.html
print("Théorie de la décision")

#SEANCE 5 DEBUT : théorie de l'échantillonnage, estimation et décision

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats # Nécessaire pour Shapiro-Wilks

#======================================
# Constantes et chemin d'accès
# Définition de la Population Mère (Théorie de l'échantillonnage)
N_POPULATION = 2185 
POP_POUR = 852
POP_CONTRE = 911
POP_SANS_OPINION = 422

# Chemins des fichiers
BASE_PATH = r"C:\Python\seance_4\src\data"
CHEMIN_ECHANTILLONS = os.path.join(BASE_PATH, r"C:\Python\seance_5\src\data\Echantillonnage-100-Echantillons.csv")
CHEMIN_TEST_1 = os.path.join(BASE_PATH, r"C:\Python\seance_5\src\data\Loi-normale-Test-1.csv")
CHEMIN_TEST_2 = os.path.join(BASE_PATH, r"C:\Python\seance_5\src\data\Loi-normale-Test-2.csv")

# Constantes Statistiques
Z_SCORE_IC = 1.96 # Z-score pour un intervalle de 95%
ALPHA = 0.05      # Seuil de signification pour le test de décision (Shapiro-Wilks)

def ouvrirUnFichier(chemin):
    """Ouvre le fichier CSV et retourne le DataFrame."""
    try:
        df = pd.read_csv(chemin)
        print(f"Fichier {os.path.basename(chemin)} chargé avec {len(df)} échantillons.")
        return df
    except FileNotFoundError:
        print(f"ERREUR : Fichier non trouvé à {chemin}")
        return pd.DataFrame()

df_echantillons = ouvrirUnFichier(CHEMIN_ECHANTILLONS)

# =====================================
# théorie de l'échantillonnage (moyennes et fréquences)
if not df_echantillons.empty:
    
    # 1. Calcul de la moyenne de chaque colonne sur les 100 échantillons
    moyenne_pour = df_echantillons['Pour'].mean()
    moyenne_contre = df_echantillons['Contre'].mean()
    moyenne_sans_opinion = df_echantillons['Sans opinion'].mean()
    
    # 2. Arrondi à l'entier le plus proche (règle round())
    moyenne_pour_arrondie = np.round(moyenne_pour, 0).astype(int)
    moyenne_contre_arrondie = np.round(moyenne_contre, 0).astype(int)
    moyenne_sans_opinion_arrondie = np.round(moyenne_sans_opinion, 0).astype(int)

    print("\nMoyennes des Échantillons (Arrondies)")
    print(f"Moyenne 'Pour' : {moyenne_pour:.4f} -> {moyenne_pour_arrondie}")
    print(f"Moyenne 'Contre' : {moyenne_contre:.4f} -> {moyenne_contre_arrondie}")
    print(f"Moyenne 'Sans opinion' : {moyenne_sans_opinion:.4f} -> {moyenne_sans_opinion_arrondie}")

    # Comparaison des Fréquences
    somme_pop = POP_POUR + POP_CONTRE + POP_SANS_OPINION
    
    # Fréquences de la population mère (théoriques)
    freq_pop_pour = np.round(POP_POUR / somme_pop, 2)
    freq_pop_contre = np.round(POP_CONTRE / somme_pop, 2)
    freq_pop_sans_opinion = np.round(POP_SANS_OPINION / somme_pop, 2)
    
    # Fréquences des échantillons (observées)
    somme_moyennes = moyenne_pour_arrondie + moyenne_contre_arrondie + moyenne_sans_opinion_arrondie
    freq_ech_pour = np.round(moyenne_pour_arrondie / somme_moyennes, 2)
    freq_ech_contre = np.round(moyenne_contre_arrondie / somme_moyennes, 2)
    freq_ech_sans_opinion = np.round(moyenne_sans_opinion_arrondie / somme_moyennes, 2)

    print("\nFréquences (Comparaison)")
    print(f"| Catégorie      | Pop. Mère (Théorique) | Échantillons (Observé) |")
    print(f"| :------------- | :-------------------- | :--------------------- |")
    print(f"| Pour           | {freq_pop_pour:.2f}                 | {freq_ech_pour:.2f}                    |")
    print(f"| Contre         | {freq_pop_contre:.2f}                 | {freq_ech_contre:.2f}                    |")
    print(f"| Sans opinion   | {freq_pop_sans_opinion:.2f}                 | {freq_ech_sans_opinion:.2f}                    |")


# =====================================
# théorie de l'estimation (intervalle de confiance)

def calculer_intervalle_confiance(frequence, taille_echantillon, z_score):
    """Calcule l'intervalle de confiance pour une fréquence."""
    
    ecart_type_frequence = np.sqrt(frequence * (1 - frequence) / taille_echantillon)
    marge_erreur = z_score * ecart_type_frequence
    borne_inf = max(0, frequence - marge_erreur)
    borne_sup = min(1, frequence + marge_erreur)
    
    return borne_inf, borne_sup

if not df_echantillons.empty:
    
    # Extraction du premier échantillon (iloc(0))
    premier_echantillon = df_echantillons.iloc[0].tolist()
    n_echantillon = sum(premier_echantillon) 
    
    ech_pour = premier_echantillon[0]
    ech_contre = premier_echantillon[1]
    ech_sans_opinion = premier_echantillon[2]

    f_ech_pour = ech_pour / n_echantillon
    f_ech_contre = ech_contre / n_echantillon
    f_ech_sans_opinion = ech_sans_opinion / n_echantillon
    
    print(f"\nIntervalles de Confiance (IC) à 95% (n={n_echantillon})")
    
    # IC pour 'Pour'
    ic_pour_inf, ic_pour_sup = calculer_intervalle_confiance(f_ech_pour, n_echantillon, Z_SCORE_IC)
    print(f"IC 95% 'Pour' (f={f_ech_pour:.4f}) : [{ic_pour_inf:.4f} ; {ic_pour_sup:.4f}]")

    # IC pour 'Contre'
    ic_contre_inf, ic_contre_sup = calculer_intervalle_confiance(f_ech_contre, n_echantillon, Z_SCORE_IC)
    print(f"IC 95% 'Contre' (f={f_ech_contre:.4f}) : [{ic_contre_inf:.4f} ; {ic_contre_sup:.4f}]")

    # IC pour 'Sans opinion'
    ic_sans_opinion_inf, ic_sans_opinion_sup = calculer_intervalle_confiance(f_ech_sans_opinion, n_echantillon, Z_SCORE_IC)
    print(f"IC 95% 'Sans opinion' (f={f_ech_sans_opinion:.4f}) : [{ic_sans_opinion_inf:.4f} ; {ic_sans_opinion_sup:.4f}]")
    
    # L'interprétation pour le rapport : comparer les IC avec les fréquences réelles (freq_pop_x).


# =======================================================
# théorie de la décision (Test de Shapiro-Wilks)

def executer_shapiro_wilks(chemin_fichier, nom_test):
    """Applique le test de Shapiro-Wilks et interprète le résultat pour la normalité."""
    try:
        
        df_test = pd.read_csv(chemin_fichier, header=None, sep=',')
        
        # Récupérer la première colonne (indice 0) et forcer la conversion en numérique
        # `errors='coerce'` remplace les chaînes (comme 'Test') par NaN
        data_brute = df_test.iloc[:, 0]
        data_numerique = pd.to_numeric(data_brute, errors='coerce').dropna() # .dropna() retire le 'Test' converti en NaN

        if data_numerique.empty:
             print(f"ERREUR : {nom_test} ne contient aucune donnée numérique valide après nettoyage.")
             return None, None
        
        # Test de Shapiro-Wilks : H0: la distribution est normale
        statistique, p_value = stats.shapiro(data_numerique)
        
        # (Reste de l'affichage et de l'interprétation)
        print(f"\nTest de Shapiro-Wilks sur {nom_test}")
        print(f"Statistique W : {statistique:.4f}")
        print(f"P-value : {p_value:.4f}")
        
        if p_value > ALPHA:
            conclusion = "NORMALE (Acceptation de H0)"
            is_normal = True
        else:
            conclusion = "PAS NORMALE (Rejet de H0)"
            is_normal = False
            
        print(f"Seuil (Alpha) : {ALPHA}")
        print(f"CONCLUSION : La distribution est considérée comme {conclusion}.")
        return is_normal, chemin_fichier

    except FileNotFoundError:
        print(f"\nERREUR : Fichier de test non trouvé à {chemin_fichier}")
        return None, None
    except Exception as e:
        print(f"\nUne erreur s'est produite lors du test sur {nom_test}: {e}")
        return None, None


# Lancer les tests
est_normale_test1, chemin_test1 = executer_shapiro_wilks(CHEMIN_TEST_1, "Loi-normale-Test-1")
est_normale_test2, chemin_test2 = executer_shapiro_wilks(CHEMIN_TEST_2, "Loi-normale-Test-2")


# ===================================
# Bonus: Identification de la distribution non-normale

chemin_non_normale = None
if est_normale_test1 is False:
    chemin_non_normale = chemin_test1
elif est_normale_test2 is False:
    chemin_non_normale = chemin_test2
    
if chemin_non_normale:
    try:
        df_non_normale = pd.read_csv(chemin_non_normale, header=None)
        data_non_normale = df_non_normale.iloc[:, 0].dropna()
        
        plt.figure(figsize=(8, 5))
        data_non_normale.hist(bins=30, density=True, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f"Bonus: Histogramme de la distribution non-normale\n({os.path.basename(chemin_non_normale)})")
        plt.xlabel("Valeur")
        plt.ylabel("Densité")
        plt.show()
        
        print("\nBONUS : Identification")
        
    except Exception as e:
        print(f"Erreur lors de la visualisation du bonus : {e}")





