import numpy as np
import matplotlib.pyplot as plt

from rate.vasicek import VasicekModel
from stochastic_process.gbm_process import GBMProcess
from market.market import Market

# === 1. Paramètres de simulation ===
n_paths = 100
n_steps = 252
dt = 1 / 252
T = n_steps * dt
rho = -0.15 # corrélation GBM / taux

# === 2. Modèle de taux : Vasicek ===
vasicek = VasicekModel(
    a=1.2,
    b=0.03,
    sigma=0.01,
    r0=0.02,
    dt=dt,
    n_steps=n_steps,
    n_paths=n_paths,
    seed=47
)

# === 3. Marché (importé) ===
market = Market(
    S0=100,
    r=0.03,       # taux de référence (inutile ici car Vasicek surpasse)
    sigma=0.2,
    dividend=0.0,
    div_type="continuous"
)

# === 4. Corrélation ===
corr_matrix = np.array([[1.0, rho],
                        [rho, 1.0]])

# === 5. Simulation GBM avec diffusion Vasicek corrélée ===
gbm = GBMProcess(
    market=market,
    dt=dt,
    n_paths=n_paths,
    n_steps=n_steps,
    t_div=None,
    rate_model=vasicek,
    correlation_matrix=corr_matrix,
    seed=47
)

paths = gbm.simulate()  # (n_paths, n_steps + 1)

# === 6. Visualisation ===
time_grid = np.linspace(0, T, n_steps + 1)

plt.figure(figsize=(10, 5))
for i in range(min(10, n_paths)):
    plt.plot(time_grid, paths[i], lw=1)
plt.title("Diffusion du sous-jacent avec taux Vasicek corrélé (ρ = {:.2f})".format(rho))
plt.xlabel("Temps")
plt.ylabel("Sous-jacent S(t)")
plt.grid(True)
plt.tight_layout()
plt.show()