from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreePortfolio
from datetime import datetime, timedelta
import numpy as np

from pricers.structured_pricer import StructuredPricer
from option.option import (
    OptionPortfolio, Call, Put,
    DigitalCall, DigitalPut,
    UpAndOutCall, DownAndOutPut,
    UpAndInCall, DownAndInPut
)

from investment_strategies.vanilla_strategies import (
    BearCallSpread,
    BullCallSpread,
    ButterflySpread,
    Straddle,
    Strap,
    Strip,
    Strangle,
    Condor,
    PutCallSpread,
    plot_strategy_payoff
)

from risk_metrics.options_backtest import Backtest
from investment_strategies.structured_strategies import (
    ReverseConvertible,
    TwinWin,
    BonusCertificate,
    CappedParticipationCertificate,
    DiscountCertificate,
    ReverseConvertibleBarrier,
    SweetAutocall
)

from rate.product import (
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    ForwardRate,
    ForwardRateAgreement,
    InterestRateSwap
)

# === 1) Chargement de la courbe de taux ===
from market.market_factory import create_market

# === 1) Définir la date de pricing et la maturité (3 ans) ===
pricing_date  = datetime(2023, 4, 25)
maturity_date = pricing_date + timedelta(days=365 * 3)

# === 2) Paramètres pour Svensson ===
sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

# === 3) Instanciation « tout‐en‐un » du Market LVMH ===
market_lvmh = create_market(
    stock         = "LVMH",
    pricing_date  = pricing_date,
    vol_source    = "implied",         # ou "historical"
    hist_window   = 252,
    curve_method  = "svensson",        # méthode de calibration
    curve_kwargs  = {"initial_guess": sv_guess},
    dcc           = "Actual/365"
)

market_lvmh.S0 = 100

# === 1) Définir la date de pricing et la maturité (3 ans) ===
pricing_date  = datetime(2025, 4, 25)
maturity_date = pricing_date + timedelta(days=365 * 3)

# === 2) Paramètres pour Svensson ===
sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

# === 3) Instanciation « tout‐en‐un » du Market LVMH ===
market_lvmh = create_market(
    stock         = "LVMH",
    pricing_date  = pricing_date,
    vol_source    = "implied",         # ou "historical"
    hist_window   = 252,
    curve_method  = "svensson",        # méthode de calibration
    curve_kwargs  = {"initial_guess": sv_guess},
    dcc           = "Actual/365",
    flat_rate     = 0.05
)

K = market_lvmh.S0*(0.9)

options = OptionPortfolio([
    Call(K, maturity_date, exercise="european"),
    Put(K, maturity_date, exercise="european"),
    ]
)

# --- Paramètres ---
n_paths = 10000
n_steps = 300
seed = 2

print("\n====== TRINOMIAL TREE PRICING ======")

engine = TreePortfolio(
    market=market_lvmh,
    option_ptf=options,
    pricing_date=pricing_date,
    n_steps=n_steps
)

price = engine.price()
print(f"Prix estimé (Trinomial Tree) : {np.round(price,4)}")

# === 4) Initialisation du StructuredPricer ===
pricer = StructuredPricer(
    market=market_lvmh,
    pricing_date=pricing_date,
    df_curve=market_lvmh.discount,
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
print("\n====== PRICING DES PRODUITS STRUCTURÉS ======\n")
for prod in products:
    price = prod.price(pricer)
    pct = price / prod.notional * 100
    print(f"{prod.name:35s} → Prix estimé : {pct:6.2f}% du notional")

# --- Date de pricing et maturité à +1 an ---
pricing_date = datetime.today()
maturity_date = pricing_date + timedelta(days=365)

# --- Strike optionnel fictif -----
K = 100

# ------ Backtest sur une option -------
c = Call(K=100,maturity=maturity_date)
shift_date = pricing_date + timedelta(days=14)

# === Backtest Setup (Monte Carlo) ===
print(f'\n --------  VaR via modèle Monte Carlo -------- ')
mce = MonteCarloEngine(market=market_lvmh, option_ptf=OptionPortfolio([c]), pricing_date=pricing_date, n_paths=10000, n_steps=300, seed=2)
backtest = Backtest(model=mce, shift_date=shift_date)

# === VaR Théorique ===
var_th = backtest.run(var_type="TH", alpha=0.05)
print(f"VaR Théorique du Portefeuille : {var_th}")

# === VaR Monte Carlo ===
var_mc = backtest.run(var_type="MC", alpha=0.05, nb_simu=100)
print(f"VaR Monte Carlo du Portefeuille : {var_mc}")

# === VaR Cornish Fisher ===
var_cf = backtest.run(var_type="CF", alpha=0.05, order=2)
print(f"VaR Cornish Fisher du Portefeuille : {var_cf}")


options = OptionPortfolio([
    Call(K, maturity_date, exercise="european"),
    Put(K, maturity_date, exercise="european"),
    DigitalCall(K, maturity_date, exercise="european", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="european", payoff=10.0),
    UpAndOutCall(K, maturity_date, barrier=150, rebate=0, exercise="european"),
    DownAndOutPut(K, maturity_date, barrier=60, rebate=0, exercise="european"),
    UpAndInCall(K, maturity_date, barrier=150, rebate=0, exercise="european"),
    DownAndInPut(K, maturity_date, barrier=60, rebate=0, exercise="european"),
])

# --- Paramètres ---
n_paths = 10000
n_steps = 300
seed = 2


print("\n====== EUROPEAN MONTE CARLO PRICING ======")

engine = MonteCarloEngine(
    market=market_lvmh,
    option_ptf=options,
    pricing_date=datetime.today(),
    n_paths=n_paths,
    n_steps=n_steps,
    seed=seed
)

price = engine.price(type="MC")
ci_low, ci_up = engine.price_confidence_interval(type="MC")

print(f"Prix estimé      : {np.round(price,4)}")
print(f"Intervalle 95%   : [{np.round(ci_low,4)}, {np.round(ci_up,4)}]")


print("\n====== AMERICAN LONGSTAFF PRICING ======")

# --- Pricing des options ---
try:
    price = engine.price(type="Longstaff")
    ci_low, ci_up = engine.price_confidence_interval(type="Longstaff")

    print(f"Prix estimé      : {np.round(price, 4)}")
    print(f"Intervalle 95%   : [{np.round(ci_low,4)}, {np.round(ci_up,4)}]")

except NotImplementedError as e:
    print(f"NotImplementedError: {e}")


# --- Liste de stratégies à tester ---
strategies = [
    BearCallSpread(strike_sell=95, strike_buy=105, pricing_date=pricing_date, maturity_date=maturity_date),
    BullCallSpread(strike_buy=95, strike_sell=105, pricing_date=pricing_date, maturity_date=maturity_date),
    ButterflySpread(strike_low=90, strike_mid=100, strike_high=110, pricing_date=pricing_date, maturity_date=maturity_date),
    Straddle(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
    Strap(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
    Strip(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
    Strangle(lower_strike=90, upper_strike=110, pricing_date=pricing_date, maturity_date=maturity_date),
    Condor(strike1=90, strike2=95, strike3=105, strike4=110, pricing_date=pricing_date, maturity_date=maturity_date),
    PutCallSpread(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
]


print("\n====== EUROPEAN VANILLA STRATEGIES PRICING ======")


# Options testables avec l'arbre trinomial (pas de barrières ici)
options_tree = OptionPortfolio([
    Call(K, maturity_date, exercise="european"),
    Put(K, maturity_date, exercise="european"),
    DigitalCall(K, maturity_date, exercise="european", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="european", payoff=10.0),
    Call(K, maturity_date, exercise="american"),
    Put(K, maturity_date, exercise="american"),
    DigitalCall(K, maturity_date, exercise="american", payoff=10.0),
    DigitalPut(K, maturity_date, exercise="american", payoff=10.0),
])

print("\n====== TRINOMIAL TREE PRICING ======")

engine = TreePortfolio(
    market=market_lvmh,
    option_ptf=options_tree,
    pricing_date=pricing_date,
    n_steps=n_steps
)

price = engine.price()
print(f"Prix estimé (Trinomial Tree) : {np.round(price,4)}")

# === Paramètres généraux ===
valuation_date = datetime(2023, 4, 25)
maturity = datetime(2028, 4, 25)
notional = 1_000_000
face_value = 1_000
freq = 2  # semestriel


# === 2) Zero Coupon Bond ===
zcb = ZeroCouponBond(
    face_value=face_value,
    pricing_date=valuation_date,
    maturity_date=maturity,
    convention_days="Actual/365"
)
print("ZCB price:", round(zcb.price(market_lvmh.discount), 2))

# === 3) Fixed Rate Bond 6% semestriel ===
frb = FixedRateBond(
    face_value=face_value,
    coupon_rate=0.06,
    pricing_date=valuation_date,
    maturity_date=maturity,
    convention_days="30/360",
    frequency=freq
)
print("Fixed Rate Bond price:", round(frb.price(market_lvmh.discount), 2))

# === 4) Floating Rate Bond ===
# Génération des taux forward pour chaque période de paiement

from market.day_count_convention import DayCountConvention

dates = frb.generate_schedule(valuation_date, maturity, freq)
dcc = DayCountConvention("Actual/365")
t_js = [dcc.year_fraction(valuation_date, d) for d in dates]
forwards = [
    market_lvmh.forward(t_js[i - 1] if i > 0 else 0.0, t_js[i])
    for i in range(len(t_js))
]

flo = FloatingRateBond(
    face_value=face_value,
    margin=0.002,
    pricing_date=valuation_date,
    maturity_date=maturity,
    forecasted_rates=forwards,
    convention_days="Actual/365",
    frequency=freq,
    multiplier=1.0
)
print("Floating Rate Bond price:", round(flo.price(market_lvmh.discount), 2))

# === 5) Taux forward discret 1->2 ans ===
start = valuation_date + timedelta(days=365)
end = valuation_date + timedelta(days=365 * 2)
fwd = ForwardRate(
    pricing_date=valuation_date,
    start_date=start,
    end_date=end,
    convention_days="Actual/365"
)
print(
    f"Forward rate {start.date()} -> {end.date()}: {fwd.value(market_lvmh.discount) * 100:.2f}%"
)

# === 6) FRA 1->2 ans ===
strike = fwd.value(market_lvmh.discount)
fra = ForwardRateAgreement(
    notional=notional,
    strike=strike,
    pricing_date=valuation_date,
    start_date=start,
    end_date=end,
    convention_days="Actual/365"
)
print("FRA MtM (payer fixe):", round(fra.mtm(market_lvmh.discount), 2), "€")

# === 7) Swap payer fixe 5 ans semestriel ===
swap = InterestRateSwap(
    notional=notional,
    fixed_rate=0.035,
    pricing_date=valuation_date,
    maturity_date=maturity,
    convention_days="30/360",
    frequency=freq,
    forecasted_rates=forwards,
    multiplier=1.0,
    margin=0.0
)
print("Swap MtM (payer fixe):", round(swap.mtm(market_lvmh.discount), 2), "€")
print(f"Swap par rate: {swap.swap_rate(market_lvmh.discount) * 100:.2f}%")