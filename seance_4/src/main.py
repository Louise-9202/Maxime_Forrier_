#coding:utf8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from scipy import stats

#https://docs.scipy.org/doc/scipy/reference/stats.html


dist_names = ['norm', 'beta', 'gamma', 'pareto', 't', 'lognorm', 'invgamma', 'invgauss',  'loggamma', 'alpha', 'chi', 'chi2', 'bradford', 'burr', 'burr12', 'cauchy', 'dweibull', 'erlang', 'expon', 'exponnorm', 'exponweib', 'exponpow', 'f', 'genpareto', 'gausshyper', 'gibrat', 'gompertz', 'gumbel_r', 'pareto', 'pearson3', 'powerlaw', 'triang', 'weibull_min', 'weibull_max', 'bernoulli', 'betabinom', 'betanbinom', 'binom', 'geom', 'hypergeom', 'logser', 'nbinom', 'poisson', 'poisson_binom', 'randint', 'zipf', 'zipfian']

print(dist_names)
# chemin du csv de la séance 4
chemin_csv = r"C:\Python\seance_4\src\main.py"

# ===========================
# Visualisation des distributions discrètes
def visualiser_distributions_discretes():
    # Définir les valeurs possibles pour l'axe des x
    x = np.arange(-5, 20)

    plt.figure(figsize=(15, 10))
    plt.suptitle("Distributions Statistiques Discrètes (PMF)", fontsize=16)

    # loi uniforme discrète
    plt.subplot(2, 3, 1)
    low, high = 1, 7 # Plage de valeurs (de 1 à 6 inclus)
    uniform_d = stats.randint(low, high)
    plt.plot(x, uniform_d.pmf(x), 'bo', ms=8, label=f'uniforme discrète ({low} à {high-1})')
    plt.vlines(x, 0, uniform_d.pmf(x), colors='b', lw=5, alpha=0.5)
    plt.title("Loi Uniforme Discrète")
    plt.legend()

    # loi binomiale
    plt.subplot(2, 3, 2)
    n, p = 10, 0.4
    binom = stats.binom(n, p)
    plt.plot(x, binom.pmf(x), 'go', ms=8, label=f'binomiale (n={n}, p={p})')
    plt.vlines(x, 0, binom.pmf(x), colors='g', lw=5, alpha=0.5)
    plt.title("Loi Binomiale")
    plt.legend()
    
    # loi de Poisson (ex: événements rares)
    plt.subplot(2, 3, 3)
    mu = 3.6 # Taux d'occurrence (lambda)
    poisson = stats.poisson(mu)
    plt.plot(x, poisson.pmf(x), 'ro', ms=8, label=f'Poisson (mu={mu})')
    plt.vlines(x, 0, poisson.pmf(x), colors='r', lw=5, alpha=0.5)
    plt.title("Loi de Poisson")
    plt.legend()

    # loi de Dirac (ou masse de probabilité au point a)
    # pas dans scipy.stats, donc modélisée
    plt.subplot(2, 3, 4)
    a = 5
    dirac_pmf = np.where(x == a, 1.0, 0.0)
    plt.plot(x, dirac_pmf, 'ko', ms=8, label=f'Dirac (a={a})')
    plt.vlines(x, 0, dirac_pmf, colors='k', lw=5, alpha=0.5)
    plt.title("Loi de Dirac (Fonction delta)")
    plt.legend()
    
    # Loi de Zipf-Mandelbrot
    # stats.zipf pour approximation simplifiée
    plt.subplot(2, 3, 5)
    a = 1.5 # Paramètre de la distribution
    zipf = stats.zipf(a)
    x_zipf = np.arange(1, 15)
    plt.plot(x_zipf, zipf.pmf(x_zipf), 'mo', ms=8, label=f'Zipf (a={a})')
    plt.vlines(x_zipf, 0, zipf.pmf(x_zipf), colors='m', lw=5, alpha=0.5)
    plt.title("Loi de Zipf (approximée)")
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Appel fonction de visualisation
visualiser_distributions_discretes()
# Fermer les figures après affichage pour libérer mémoire graphique
plt.close('all')

# ===============================
# QUESTION 1 - Visualisation des distributions continues
def visualiser_distributions_continues():
    # Définition de l'espace pour l'axe des x
    x_cont = np.linspace(-5, 10, 500)

    plt.figure(figsize=(15, 10))
    plt.suptitle("Distributions Statistiques Continues", fontsize=16)

    # Loi normale (Gaussienne)
    plt.subplot(2, 3, 1)
    mu, sigma = 0, 1 # Moyenne et écart type
    norm = stats.norm(loc=mu, scale=sigma)
    plt.plot(x_cont, norm.pdf(x_cont), label=f'Normale (mu={mu}, sigma={sigma})')
    plt.title("Loi Normale")
    plt.legend()

    # Loi log-normale
    plt.subplot(2, 3, 2)
    s = 0.9 # Paramètre de forme
    lognorm = stats.lognorm(s)
    x_lognorm = np.linspace(lognorm.ppf(0.01), lognorm.ppf(0.99), 100)
    plt.plot(x_lognorm, lognorm.pdf(x_lognorm), label=f'Log-Normale (s={s})')
    plt.title("Loi Log-Normale")
    plt.legend()

    # Loi uniforme continue
    plt.subplot(2, 3, 3)
    a, b = -2, 4
    uniform_c = stats.uniform(loc=a, scale=b-a) # scale = b-a
    plt.plot(x_cont, uniform_c.pdf(x_cont), label=f'Uniforme ({a} à {b})')
    plt.title("Loi Uniforme Continue")
    plt.legend()

    # Loi du Chi-carré (Chi²)
    plt.subplot(2, 3, 4)
    df = 5 # (degrés libertés)
    chi2 = stats.chi2(df)
    x_chi2 = np.linspace(chi2.ppf(0.01), chi2.ppf(0.99), 100)
    plt.plot(x_chi2, chi2.pdf(x_chi2), label=f'Chi² (df={df})')
    plt.title("Loi du Chi-carré")
    plt.legend()
    
    # Loi de Pareto
    plt.subplot(2, 3, 5)
    b_pareto = 2
    pareto = stats.pareto(b_pareto)
    x_pareto = np.linspace(pareto.ppf(0.01), pareto.ppf(0.99), 100)
    plt.plot(x_pareto, pareto.pdf(x_pareto), label=f'Pareto (b={b_pareto})')
    plt.title("Loi de Pareto")
    plt.legend()

    # loi de Poisson est discrète donc exclue ici de la visualisation continue

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Appel fonction de visualisation
visualiser_distributions_continues()

# Fermer les figures après affichage pour libérer mémoire graphique
plt.close('all')

# ==============================
# QUESTION 2 - CALCULS STATISTIQUES

def calculer_stats(distribution_obj, nom: str):
    """Calcule et affiche moyenne et écart type d'un objet de distribution scipy."""
    m, v, _, _ = distribution_obj.stats(moments='mvsk') # m=moyenne, v=variance
    std_dev = np.sqrt(v)
    
    print(f"▶ {nom}:")
    print(f"  - Moyenne (μ) : {m:.4f}")
    print(f"  - Écart type (σ): {std_dev:.4f}")
    return m, std_dev

print("\nMoyenne et Écart Type des Distributions (scipy.stats)")

# DISCRETES
print("\nDistributions Discrètes")
# Uniforme Discrète
uniform_d = stats.randint(low=1, high=7)
calculer_stats(uniform_d, "Uniforme Discrète")

# Binomiale (n=10, p=0.4)
binom = stats.binom(n=10, p=0.4)
calculer_stats(binom, "Binomiale")

# Poisson (mu=3.6)
poisson_d = stats.poisson(mu=3.6)
calculer_stats(poisson_d, "Poisson (Discrète)")

# Dirac (pour masse de probabilité au point a=5)
# loi de Dirac au point 'a' a une moyenne de 'a' et un écart type de 0.
print("▶ Dirac (a=5):")
print("  - Moyenne (μ) : 5.0000")
print("  - Écart type (σ): 0.0000")

# Zipf (a=1.5)
zipf = stats.zipf(a=1.5) 
calculer_stats(zipf, "Zipf (approximée)")

# CONTINUES
print("\nDistributions Continues")
# Poisson (n'est pas une loi continue donc ignorée)

# Normale (mu=0, sigma=1)
norm = stats.norm(loc=0, scale=1)
calculer_stats(norm, "Normale")

# Log-Normale (s=0.9)
lognorm = stats.lognorm(s=0.9)
calculer_stats(lognorm, "Log-Normale")

# Uniforme Continue (a=-2, b=4)
uniform_c = stats.uniform(loc=-2, scale=4-(-2)) # scale = b-a
calculer_stats(uniform_c, "Uniforme Continue")

# Chi-carré (df=5)
chi2 = stats.chi2(df=5)
calculer_stats(chi2, "Chi-carré")

# Pareto (b=2)
pareto = stats.pareto(b=2)
calculer_stats(pareto, "Pareto")

