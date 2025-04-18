from rate.nelson_siegel import NelsonSiegelModel
from volatility.vol_surface import OptionVolSurface
from stochastic_process.gbm_process import GBMProcess
from data_management.manage_bloomberg_data import OptionDataParser
import numpy as np
import pandas as pd

# Données brutes
np.random.seed(123)
data = pd.read_excel("data_taux/RateCurve_temp.xlsx")
taus_raw = data['Matu'].values
observed_yields = data['Rate'].values

# data option
file_path = "data_options/options_data_TSLA 2.xlsx"
df_options = OptionDataParser.prepare_option_data(file_path)

# Création de la surface de volatilité
vol_surface = OptionVolSurface(df_options)

# 1. Calibration de la courbe des taux
ns_model = NelsonSiegelModel(beta0=4.0, beta1=-1.0, beta2=0.5, lambda1=2)
ns_model.calibrate(taus_raw, observed_yields, initial_guess=[4.0, -1.0, 0.5, 2])
ns_model.plot_fit(taus_raw, observed_yields)

# 2. Extraction du taux pour une maturité T (pour actualiser)
T = 1.0  # maturité de l'option, par exemple 1 an
r_T = ns_model.yield_curve(T)

strikes, vols = vol_surface.get_vol_smile(maturity=T)
# Pour simplifier, supposons que vous obtenez la volatilité correspondant à votre strike K
K = 100
sigma_T = np.interp(K, strikes, vols)

# 4. Simulation de l'actif sous-jacent
bs_process = GBMProcess(S0=100, r=r_T, sigma=sigma_T)
time_grid, paths = bs_process.simulate(T=T, n_paths=10000, n_steps=252)

# 5. Calcul du payoff et estimation du prix
S_T = paths[-1]
K = 100  # Strike de l'option
payoffs_call = np.maximum(S_T - K, 0)
payoffs_put = np.maximum(K - S_T, 0)

# Actualisation des payoffs
discount_factor = np.exp(-r_T * T)
price_call = discount_factor * np.mean(payoffs_call)
price_put  = discount_factor * np.mean(payoffs_put)

print("Prix du Call:", price_call)
print("Prix du Put:", price_put)
