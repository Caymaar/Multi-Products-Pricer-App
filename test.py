from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreeModel
from market.market import Market
from option.option import (
    Call, Put,
    DigitalCall, DigitalPut,
    UpAndOutCall, DownAndOutPut,
    UpAndInCall, DownAndInPut
)
from investment_strategies.vanilla_strategy import (
    BearCallSpread, BullCallSpread,
    ButterflySpread, Straddle,
    Strap, Strip, Strangle, Condor,
    PutCallSpread
)

from datetime import datetime, timedelta

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

# --- Liste d'options européennes à tester ---
options = [
    Call(K, maturity_date, exercise="european"),
    Put(K, maturity_date, exercise="european"),
    DigitalCall(K, maturity_date, exercise="european", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="european", payoff=10.0),
    UpAndOutCall(K,maturity_date, barrier=120, rebate=0, exercise="european"),
    DownAndOutPut(K, maturity_date, barrier=80, rebate=0, exercise="european"),
    UpAndInCall(K, maturity_date, barrier=120, rebate=0, exercise="european"),
    DownAndInPut(K, maturity_date, barrier=80, rebate=0, exercise="european"),
]

# --- Paramètres ---
n_paths = 100000
n_steps = 300
seed = 2

print("\n====== EUROPEAN MONTE CARLO PRICING ======")

# --- Pricing des options ---
for opt in options:
    print(f"\n--- {opt.__class__.__name__} ---")

    engine = MonteCarloEngine(
        market=market,
        option=opt,
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
    UpAndOutCall(K,maturity_date, barrier=120, rebate=0, exercise="american"),
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


