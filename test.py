from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreeModel
from datetime import datetime, timedelta
import pandas as pd
from market.market import Market
from pricers.structured_pricer import StructuredPricer
from option.option import (
    OptionPortfolio, Call, Put,
    DigitalCall, DigitalPut,
    UpAndOutCall, DownAndOutPut,
    UpAndInCall, DownAndInPut
)
from investment_strategies.structured_strategies import (
    ReverseConvertible,
    TwinWin,
    BonusCertificate,
    CappedParticipationCertificate,
    DiscountCertificate,
    ReverseConvertibleBarrier,
    SweetAutocall
)

# === 1) Chargement de la courbe de taux ===
data = pd.read_excel("./data_taux/RateCurve_temp.xlsx")
maturities = data['Matu'].values
yields = data['Rate'].values / 100.0

zc_method = "interpolation"
zc_args = (maturities, yields)

# === 2) Marché actions pour les options ===
S0, r_eq, sigma, div = 100, 0.05, 0.2, 0.0
market_stock = Market(S0, r_eq, sigma, div, div_type="continuous")

# === 3) Dates de pricing et maturité (3 ans) ===
pricing_date  = datetime.today()
maturity_date = pricing_date + timedelta(days=365 * 3)

# === 4) Initialisation du StructuredPricer ===
pricer = StructuredPricer(
    market=market_stock,
    pricing_date=pricing_date,
    zc_method=zc_method,
    zc_args=zc_args,
    n_paths=10_000,
    n_steps=300,
    seed=42,
    compute_antithetic=True
)

# === 5) Définition des dates d’observation et des taux de coupon ===
obs_dates = [
    pricing_date + timedelta(days=365 * i)
    for i in (1, 2, 3)
]
coupon_rates = [0.05, 0.05, 0.05]  # 5 % par an

# === 6) Instanciation des produits structurés ===
products = [
    SweetAutocall(
        obs_dates=obs_dates,
        coupon_rates=coupon_rates,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        coupon_barrier=0.8,           # 80 % pour coupon
        call_barrier=1.1,             # 110 % pour autocall
        protection_barrier=0.8,       # 80 % pour protection à maturité
        notional=1000.0,
        convention_days="Actual/365"
    ),
    TwinWin(
        K=100,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        PDO_barrier=80,
        CUO_barrier=120,
        notional=1000.0
    ),
    ReverseConvertible(
        K=100,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        notional=1000.0
    ),
    BonusCertificate(
        K=100,
        barrier=80,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        notional=1000.0
    ),
    CappedParticipationCertificate(
        K=100,
        cap=120,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        notional=1000.0
    ),
    DiscountCertificate(
        K=100,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        notional=1000.0
    ),
    ReverseConvertibleBarrier(
        K=100,
        barrier=80,
        pricing_date=pricing_date,
        maturity_date=maturity_date,
        notional=1000.0
    )
]

# === 7) Pricing et affichage ===
from investment_strategies.structured_strategies import Autocallable

print("\n====== PRICING DES PRODUITS STRUCTURÉS ======\n")
for prod in products:
    price = prod.price(pricer)
    if isinstance(prod, Autocallable):
        pct = price
    else:
        pct = price / prod.notional * 100
    print(f"{prod.name:35s} → Prix estimé : {pct:6.2f}% du notional")

# --- Date de pricing et maturité à +1 an ---
pricing_date = datetime.today()
maturity_date = pricing_date + timedelta(days=365)

# --- Paramètres marché ---
S0 = 100
r = 0.05
sigma = 0.2
div = 0.0
K = 100

market = Market(S0, r, sigma, div, div_type="continuous")

options = OptionPortfolio([
    Call(K, maturity_date, exercise="european"),
    Put(K, maturity_date, exercise="european"),
    DigitalCall(K, maturity_date, exercise="european", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="european", payoff=10.0),
    UpAndOutCall(K, maturity_date, barrier=120, rebate=0, exercise="european"),
    DownAndOutPut(K, maturity_date, barrier=80, rebate=0, exercise="european"),
    UpAndInCall(K, maturity_date, barrier=120, rebate=0, exercise="european"),
    DownAndInPut(K, maturity_date, barrier=80, rebate=0, exercise="european"),
])
# --- Paramètres ---
n_paths = 10000
n_steps = 300
seed = 2

print("\n====== EUROPEAN MONTE CARLO PRICING ======")

engine = MonteCarloEngine(
    market=market,
    option_ptf=options,
    pricing_date=datetime.today(),
    n_paths=n_paths,
    n_steps=n_steps,
    seed=seed
)

price = engine.price(type="MC")
ci_low, ci_up = engine.price_confidence_interval(type="MC")

print(f"Prix estimé      : {price:.4f}")
print(f"Intervalle 95%   : [{ci_low:.4f}, {ci_up:.4f}]")

# Options testables avec l'arbre trinomial (pas de barrières ici)
options_tree = [
    Call(K, maturity_date, exercise="european"),
    Put(K, maturity_date, exercise="european"),
    DigitalCall(K, maturity_date, exercise="european", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="european", payoff=10.0),
    Call(K, maturity_date, exercise="american"),
    Put(K, maturity_date, exercise="american"),
    DigitalCall(K, maturity_date, exercise="american", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="american", payoff=10.0),
]

print("\n====== TRINOMIAL TREE PRICING ======")

for opt in options_tree:
    print(f"\n--- {opt.__class__.__name__} ({opt.exercise}) ---")

    engine = TreeModel(
        market=market,
        option=opt,
        pricing_date=pricing_date,
        n_steps=n_steps
    )

    price = engine.price()
    print(f"Prix estimé (Trinomial Tree) : {price:.4f}")

# --- Liste d'options américaines à tester ---
options = [
    Call(K, maturity_date, exercise="american"),
    Put(K, maturity_date, exercise="american"),
    DigitalCall(K, maturity_date, exercise="american", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="american", payoff=10.0),
    UpAndOutCall(K, maturity_date, barrier=120, rebate=0, exercise="american"),
    DownAndOutPut(K, maturity_date, barrier=80, rebate=0, exercise="american"),
    UpAndInCall(K, maturity_date, barrier=120, rebate=0, exercise="american"),
    DownAndInPut(K, maturity_date, barrier=80, rebate=0, exercise="american"),
]

print("\n====== AMERICAN LONGSTAFF PRICING ======")

# --- Pricing des options ---
for opt in options:
    print(f"\n--- {opt.__class__.__name__} (American) ---")

    engine = MonteCarloEngine(
        market=market,
        option=opt,
        pricing_date=pricing_date,
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed
    )

    try:
        price = engine.price(type="Longstaff")
        ci_low, ci_up = engine.price_confidence_interval(type="Longstaff")

        print(f"Prix estimé      : {price:.4f}")
        print(f"Intervalle 95%   : [{ci_low:.4f}, {ci_up:.4f}]")

    except NotImplementedError as e:
        print(f"NotImplementedError: {e}")
