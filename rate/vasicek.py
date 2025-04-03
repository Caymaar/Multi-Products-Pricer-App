import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# ------------------------- Calibration Vasicek par la méthode Euler–Maruyama -------------------------


def simulate_vasicek(a, b, sigma, r0, dt, N):
    """
    Simule une trajectoire du modèle de Vasicek.

    Paramètres:
        a     : vitesse de réversion
        b     : taux de long terme
        sigma : volatilité
        r0    : taux initial
        dt    : pas de temps
        N     : nombre de pas
    Retourne:
        r     : trajectoire simulée des taux d'intérêt
    """
    r = np.zeros(N)
    r[0] = r0
    for i in range(1, N):
        r[i] = r[i - 1] + a * (b - r[i - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
    return r

def simulate_vasicek_paths(a, b, sigma, r0, T, dt, n_paths):
    """
    Simule n_paths trajectoires du modèle de Vasicek en utilisant la méthode Euler–Maruyama.

    Paramètres:
        a       : vitesse de réversion
        b       : taux de long terme
        sigma   : volatilité
        r0      : taux initial
        T       : horizon de temps (en années)
        dt      : pas de temps (en années)
        n_paths : nombre de trajectoires à simuler

    Retourne:
        r : tableau NumPy de dimensions (n_paths, N)
            où N = int(T/dt). Chaque ligne correspond à une trajectoire.
    """
    N = int(T / dt)            # nombre de pas de temps
    r = np.zeros((n_paths, N)) # pour stocker toutes les trajectoires

    for i in range(n_paths):
        r[i, 0] = r0
        for t in range(1, N):
            # Discrétisation Euler–Maruyama
            r[i, t] = r[i, t-1] + a*(b - r[i, t-1])*dt + sigma*np.sqrt(dt)*np.random.randn()

    return r

# Calibration par moindres carrés (Least Squares)
def calibrate_vasicek_least_squares(r, dt):
    """
    Calibre les paramètres du modèle de Vasicek à partir d'une série temporelle r.

    On considère le modèle discret :
        r_{t+1} = c + d * r_t + erreur,
    avec c = a·b·dt et d = 1 - a·dt.

    Paramètres:
        r  : série chronologique des taux d'intérêt simulés (numpy array)
        dt : pas de temps
    Retourne:
        a_est    : estimation de la vitesse de réversion a
        b_est    : estimation du taux de long terme b
        sigma_est: estimation de la volatilité sigma
    """
    r_t = r[:-1]
    r_tp1 = r[1:]

    # Construction de la matrice X pour la régression linéaire
    X = np.vstack([np.ones(len(r_t)), r_t]).T
    # Résolution de l'équation X * beta = r_tp1 par moindres carrés
    beta, _, _, _ = np.linalg.lstsq(X, r_tp1, rcond=None)
    c, d = beta

    # Récupération des paramètres du modèle
    a_est = (1 - d) / dt
    b_est = c / (a_est * dt)

    # Estimation de sigma à partir des résidus :
    # L'erreur théorique est sigma*sqrt(dt)*ε, donc sigma_est = std(residus)/sqrt(dt)
    resid = r_tp1 - (c + d * r_t)
    sigma_est = np.std(resid, ddof=1) / np.sqrt(dt)

    return a_est, b_est, sigma_est


# Calibration des paramètres du modèle de Vasicek via libor où sofr


DATA_PATH = os.path.dirname(__file__) + "/../data_taux"
# df = pd.read_csv(f"{DATA_PATH}/libor.csv", parse_dates=["DATES"], dayfirst=True)
df = pd.read_csv(f"{DATA_PATH}/sofr_data.csv", parse_dates=["DATES"], dayfirst=True)

# Tri par date croissante (important pour que r_{t+1} suive r_t)
df.sort_values("DATES", inplace=True)
df.reset_index(drop=True, inplace=True)
# r = df["LIBOR 3M ICE"].values
r = df["SOFR (O/N)"].values

# -----------------------------
# 2. Définition du pas de temps
# -----------------------------
# Si vous supposez 252 jours de marché par an :
dt = 1.0 / 252

a_est, b_est, sigma_est = calibrate_vasicek_least_squares(r, dt)

print(f"Paramètres calibrés : a = {a_est:.4f}, b = {b_est:.4f}, sigma = {sigma_est:.4f}")

# -----------------------------
# Simulation des chemins
# -----------------------------
a = a_est        # vitesse de réversion
b = b_est       # taux de long terme
sigma = sigma_est   # volatilité
r0 = 1    # taux initial %
T = 5.0        # horizon total de 5 ans
dt = 1/252     # pas de temps (ex: quotidien = 1/252)
n_paths = 10   # nombre de trajectoires à simuler

# Simulation de plusieurs trajectoires
paths = simulate_vasicek_paths(a, b, sigma, r0, T, dt, n_paths)
N = paths.shape[1]
time_grid = np.linspace(0, T, N)


plt.figure(figsize=(10, 6))
for i in range(n_paths):
    plt.plot(time_grid, paths[i, :], lw=1, alpha=0.7)

mean_path = np.mean(paths, axis=0)
plt.plot(time_grid, mean_path, color='k', lw=2, label='Moyenne')
plt.axhline(b, color='r', ls='--', label='Moyenne long terme (b)')

plt.title("Vasicek Model - Simulated Short Rate Paths")
plt.xlabel("Temps (années)")
plt.ylabel("Taux d'intérêt")
plt.legend()
plt.show()
