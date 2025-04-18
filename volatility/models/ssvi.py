import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize, curve_fit

# === PATH TO DATA ===
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data_options"))
FILE = os.path.join(DATA_PATH, "clean_data_SPX.xlsx")

# === ATM Volatility Model ===
def theta_model(t, kappa, nu_inf, nu_0):
    with np.errstate(divide="ignore", invalid="ignore"):
        theta = ((1 - np.exp(-kappa * t)) / (kappa * t)) * (nu_inf - nu_0) + nu_0
    return theta


# === SSVI Model ===
def ssvi_total_var(k, t, theta_t, rho, eta, lambd):
    phi_t = eta / theta_t**lambd
    term = phi_t * k + rho
    w = 0.5 * theta_t * (1 + rho * phi_t * k + np.sqrt(term**2 + 1 - rho**2))
    return w


def ssvi_implied_vol(k, t, theta_t, rho, eta, lambd):
    total_var = ssvi_total_var(k, t, theta_t, rho, eta, lambd)
    return np.sqrt(np.maximum(total_var / t, 0))


# === Calibration Functions ===

def get_atm_vols(df):
    grouped = df.groupby("Maturity")
    atm_data = []
    for t, group in grouped:
        idx = (group["Strike"] - group["Spot"]).abs().idxmin()
        row = group.loc[idx]
        k = np.log(row["Strike"] / row["Spot"])
        if abs(k) < 0.05:
            atm_data.append((t, row["ImpliedVol"]))
    atm_df = pd.DataFrame(atm_data, columns=["Maturity", "ImpliedVol"])
    atm_df = atm_df.dropna()
    print(atm_df)
    print("θ_obs =", (atm_df["ImpliedVol"] ** 2 * atm_df["Maturity"]).values)
    return atm_df


def calibrate_theta(atm_df):
    t = atm_df["Maturity"].values
    theta_obs = (atm_df["ImpliedVol"].values / 100) ** 2 * t

    def objective(t, kappa, nu_0, nu_inf):
        return theta_model(t, kappa, nu_0, nu_inf)

    popt, _ = curve_fit(objective, t, theta_obs, bounds=([1e-4, 1e-6, 1e-6], [10, 5, 5]))
    return popt


def calibrate_ssvi(df, theta_func):
    k_all, t_all, iv_obs = [], [], []

    for _, row in df.iterrows():
        try:
            t = float(row["Maturity"])
            k = np.log(row["Strike"] / row["Spot"])
            iv = row["ImpliedVol"]
            k_all.append(k)
            t_all.append(t)
            iv_obs.append(iv)
        except:
            continue

    k_all, t_all, iv_obs = map(np.array, (k_all, t_all, iv_obs))

    def obj(params):
        rho, eta, lambd = params
        if not (-1 < rho < 1 and eta > 0 and lambd > 0):
            return 1e6
        theta_t = theta_func(t_all)
        model_iv = ssvi_implied_vol(k_all, t_all, theta_t, rho, eta, lambd)
        return np.mean((model_iv - iv_obs) ** 2)

    res = minimize(obj, x0=[-0.1, 0.5, 0.5],
                   bounds=[(-0.999, 0.999), (1e-5, 10), (1e-3, 5)],
                   method="L-BFGS-B")
    return res.x


# === Plotting ===

def plot_theta_fit(atm_df, theta_func):
    t_grid = np.linspace(atm_df["Maturity"].min(), atm_df["Maturity"].max(), 100)
    theta_fit = theta_func(t_grid)
    plt.figure(figsize=(8, 4))
    plt.plot(atm_df["Maturity"], atm_df["ImpliedVol"] ** 2 * atm_df["Maturity"], 'o', label="Observed θ")
    plt.plot(t_grid, theta_fit, '-', label="Fitted θ(t)")
    plt.xlabel("Maturity")
    plt.ylabel("Total Variance θ(t)")
    plt.title("ATM Total Variance Fit")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_ssvi_surface(df, theta_func, rho, eta, lambd):
    k_vals = np.linspace(-0.4, 0.4, 50)
    t_vals = np.linspace(df["Maturity"].min(), df["Maturity"].max(), 50)
    K, T = np.meshgrid(k_vals, t_vals)
    Theta = theta_func(T)
    IV = ssvi_implied_vol(K, T, Theta, rho, eta, lambd)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(K, T, IV, cmap='viridis', alpha=0.9)
    ax.set_xlabel("Log-moneyness k")
    ax.set_ylabel("Maturity t")
    ax.set_zlabel("Implied Volatility")
    ax.set_title("SSVI Implied Volatility Surface")
    plt.tight_layout()
    plt.show()


def reshape_market_matrix(raw_df):
    # La première colonne contient les maturités (en jours ou sous forme 1W, 1M, etc.)
    maturities_raw = raw_df.iloc[:, 0].copy()
    raw_df = raw_df.drop(columns=raw_df.columns[0])

    # Extraction des strikes depuis les noms de colonnes
    try:
        strikes = raw_df.columns.astype(float)
    except:
        raise ValueError("Les noms de colonnes ne semblent pas être des strikes numériques.")

    # Conversion des maturités en années si elles sont de type texte (1W, 2M, etc.)
    def parse_maturity(val):
        if isinstance(val, str):
            val = val.strip().upper()
            if "W" in val:
                return int(val.replace("W", "")) / 52
            elif "M" in val:
                return int(val.replace("M", "")) / 12
            elif "Y" in val:
                return int(val.replace("Y", ""))
        return float(val)

    maturities = np.array([parse_maturity(x) for x in maturities_raw])

    # Création du DataFrame long
    data = []
    for i, row in raw_df.iterrows():
        for j, vol in enumerate(row):
            data.append({
                "Maturity": maturities[i],
                "Strike": strikes[j],
                "ImpliedVol": vol
            })

    return pd.DataFrame(data).dropna()

# === MAIN ===

def main():
    print("Loading matrix data...")
    raw_df = pd.read_excel(FILE)

    df = reshape_market_matrix(raw_df)

    # A ajuster
    SPOT = df['Strike'].mean()  # à adapter !
    df["Spot"] = SPOT

    print("Extracting ATM vols...")
    atm_df = get_atm_vols(df)

    print("Calibrating θ(t)...")
    kappa, nu_0, nu_inf = calibrate_theta(atm_df)
    print(f"θ(t) params: κ={kappa:.4f}, ν₀={nu_0:.4f}, ν∞={nu_inf:.4f}")

    def theta_func(t):
        return theta_model(np.array(t), kappa, nu_0, nu_inf)

    plot_theta_fit(atm_df, theta_func)

    print("Calibrating SSVI parameters...")
    rho, eta, lambd = calibrate_ssvi(df, theta_func)
    print(f"SSVI params: ρ={rho:.4f}, η={eta:.4f}, λ={lambd:.4f}")

    print("Plotting SSVI surface...")
    plot_ssvi_surface(df, theta_func, rho, eta, lambd)


if __name__ == "__main__":
    main()
