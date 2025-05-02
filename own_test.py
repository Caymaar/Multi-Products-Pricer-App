from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreePortfolio
from pricers.bs_pricer import BSPortfolio
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np

from pricers.structured_pricer import StructuredPricer
from investment_strategies.structured_strategies import (
    ReverseConvertible,
    TwinWin,
    BonusCertificate,
    CappedParticipationCertificate,
    DiscountCertificate,
    ReverseConvertibleBarrier,
    SweetAutocall
)

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

from rate.product import (
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    ForwardRate,
    ForwardRateAgreement,
    InterestRateSwap
)
from risk_metrics.rate_product_sensitivity import (
    ZeroCouponSensitivity,
    FixedRateBondSensitivity,
    FloatingRateBondSensitivity,
    InterestRateSwapSensitivity,
)

from risk_metrics.options_backtest import Backtest
from risk_metrics.rate_product_sensitivity import dv01, duration, convexity

# === 1) Chargement de la courbe de taux ===
from market.market_factory import create_market

# === 1) Définir la date de pricing et la maturité (5 ans) ===
pricing_date  = datetime(2023, 4, 25)
maturity_date = datetime(2028, 4, 25)

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
    dcc           = "Actual/Actual",
)

#market_lvmh.div_date = datetime(2026,1,1) # à ajouter dans le create_market si demandé par l'user
#market_lvmh.div_type = "discrete" # à ajouter dans le create_market si demandé par l'user
#market_lvmh.dividend = 15 # à ajouter dans le create_market si demandé par l'user

# strike « out‐of‐the‐money » (ici 90% de S0)
K = market_lvmh.S0 * 0.9

# ------ Backtest sur une option -------
c = Call(K=K,maturity=maturity_date)
shift_date = pricing_date + timedelta(days=30)

# === Backtest Setup (Black Scholes) ===
bse = BSPortfolio(market=market_lvmh, option_ptf=OptionPortfolio([c]), pricing_date=pricing_date)
backtest = Backtest(model=bse, shift_date=shift_date)

# === VaR Théorique ===
var_th = backtest.run(var_type="TH", alpha=0.05)
print(f"VaR Théorique du Straddle : {var_th}")

# === VaR Monte Carlo ===
var_mc = backtest.run(var_type="MC", alpha=0.05, nb_simu=1000)
print(f"VaR Monte Carlo du Straddle : {var_mc}")

# === VaR Cornish Fisher ===
var_cf = backtest.run(var_type="CF", alpha=0.05, order=2)
print(f"VaR Cornish Fisher du Straddle: {var_cf}")

# === Backtest Setup (Trinomial) ===
print(f'\n --------  VaR via modèle Trinomial -------- ')
te = TreePortfolio(market=market_lvmh, option_ptf=OptionPortfolio([c]), pricing_date=pricing_date, n_steps=300)
backtest = Backtest(model=te, shift_date=shift_date)

# === VaR Théorique ===
var_th = backtest.run(var_type="TH", alpha=0.05)
print(f"VaR Théorique du Portefeuille : {var_th}")

# === VaR Monte Carlo ===
var_mc = backtest.run(var_type="MC", alpha=0.05, nb_simu=100)
print(f"VaR Monte Carlo du Portefeuille : {var_mc}")

# === VaR Cornish Fisher ===
var_cf = backtest.run(var_type="CF", alpha=0.05, order=2)
print(f"VaR Cornish Fisher du Portefeuille : {var_cf}")

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

# barrière “up” à 120% de S0,
# barrière “down” à 80% de S0
barrier_up   = market_lvmh.S0 * 1.2
barrier_down = market_lvmh.S0 * 0.8

# === 2) StructuredPricer ===
pricer = StructuredPricer(
    market           = market_lvmh,
    pricing_date     = pricing_date,
    df_curve         = market_lvmh.discount,
    maturity_date    = maturity_date,
    n_paths          = 10_000,
    n_steps          = 300,
    seed             = 2,
    compute_antithetic=True
)

# === 3) Obs dates et coupons pour Sweet Autocall ===
obs_dates   = [pricing_date + timedelta(days=365*i) for i in (1,2,3,4,5)]
coupon_rates = [0.05, 0.05, 0.05, 0.05, 0.05]  # 5% par an

# === 4) Instanciation des produits structurés ===
products = [
    SweetAutocall(
        freq               = "Annuel",
        coupon_rate        = 0.05,
        pricing_date       = pricing_date,
        maturity_date      = maturity_date,
        coupon_barrier     = 0.8,           # 80% de S0
        call_barrier       = 1.1,           # 110% de S0
        protection_barrier = 0.8,           # 80% de S0
        notional           = 1_000.0,
        convention_days    = "Actual/365"
    ),
    TwinWin(
        K                 = K,
        pricing_date      = pricing_date,
        maturity_date     = maturity_date,
        PDO_barrier       = barrier_down,   # 80% de S0
        CUO_barrier       = barrier_up,     # 120% de S0
        notional          = 1_000.0
    ),
    ReverseConvertible(
        K             = K,
        pricing_date  = pricing_date,
        maturity_date = maturity_date,
        notional      = 1_000.0
    ),
    BonusCertificate(
        K             = K,
        barrier       = barrier_down,       # 80% de S0
        pricing_date  = pricing_date,
        maturity_date = maturity_date,
        notional      = 1_000.0
    ),
    CappedParticipationCertificate(
        K             = K,
        cap           = barrier_up,         # 120% de S0
        pricing_date  = pricing_date,
        maturity_date = maturity_date,
        notional      = 1_000.0
    ),
    DiscountCertificate(
        K             = K,
        pricing_date  = pricing_date,
        maturity_date = maturity_date,
        notional      = 1_000.0
    ),
    ReverseConvertibleBarrier(
        K             = K,
        barrier       = barrier_down,
        pricing_date  = pricing_date,
        maturity_date = maturity_date,
        notional      = 1_000.0
    )
]

# === 5) Pricing et affichage ===
print("\n====== PRICING DES PRODUITS STRUCTURÉS ======\n")
for prod in products:
    price = prod.price(pricer)
    pct   = price / prod.notional * 100
    print(f"{prod.name:35s} → {pct:6.2f}% du notional")
    # si c'est un produit de taux, on peut aussi afficher ses sensi :
    if isinstance(prod, (ZeroCouponBond, FixedRateBond, FloatingRateBond, InterestRateSwap)):
        print(f"   · DV01      = {dv01(prod, market_lvmh.discount):.6f}")
        print(f"   · Duration  = {duration(prod, market_lvmh.discount):.6f}")
        print(f"   · Convexity = {convexity(prod, market_lvmh.discount):.6f}")
    print()

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

# --- Paramètres ---
n_paths = 10000
n_steps = 300
seed = 2

print("\n====== TRINOMIAL TREE PRICING ======")

engine = TreePortfolio(
    market=market_lvmh,
    option_ptf=options_tree,
    pricing_date=pricing_date,
    n_steps=n_steps
)

price = engine.price()
print(f"Prix estimé (Trinomial Tree) : {np.round(price,4)}")

options = OptionPortfolio([
    Call(         K, maturity_date, exercise="european"                ),
    Put(          K, maturity_date, exercise="european"                ),
    DigitalCall(  K, maturity_date, exercise="european", payoff=10.0   ),
    DigitalPut(   K, maturity_date, exercise="european", payoff=10.0   ),

    # barrier‐options : barrère au‐dessus de S0 pour les “up”, en dessous pour les “down”
    UpAndOutCall( K, maturity_date, barrier=barrier_up,   rebate=0, exercise="european" ),
    UpAndInCall(  K, maturity_date, barrier=barrier_up,   rebate=0, exercise="european" ),
    DownAndOutPut(K, maturity_date, barrier=barrier_down, rebate=0, exercise="european" ),
    DownAndInPut( K, maturity_date, barrier=barrier_down, rebate=0, exercise="european" ),
])

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

# === Paramètres généraux ===
notional = 1_000_000
face_value = 1_000
freq = 2  # semestriel

df_curve = market_lvmh.discount
fwd_curve = market_lvmh.forward

# === 2) Zero Coupon Bond ===
zcb = ZeroCouponBond(
    face_value=face_value,
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    convention_days="Actual/365"
)
zcb_sens  = ZeroCouponSensitivity(zcb, df_curve)
print("ZCB price:", round(zcb.price(df_curve), 2))
print("         · DV01       :", round(zcb_sens.dv01(), 6))
print("         · Duration   :", round(zcb_sens.macaulay_duration(), 6))
print("         · Convexity :", round(zcb_sens.convexity(), 6))

# === 3) Fixed Rate Bond 6% semestriel ===
frb = FixedRateBond(
    face_value=face_value,
    coupon_rate=0.06,
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    convention_days="30/360",
    frequency=freq
)
frb_price = frb.price(df_curve)
frb_sens  = FixedRateBondSensitivity(frb, df_curve)
print("FRB 6%   · Price      :", round(frb_price, 2))
print("         · DV01       :", round(frb_sens.dv01(), 6))
print("         · Duration   :", round(frb_sens.macaulay_duration(), 6))
print("         · Convexity :", round(frb_sens.convexity(), 6))
print()

flo = FloatingRateBond(
    face_value=face_value,
    margin=0.002,
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    forward_curve=fwd_curve,
    convention_days="Actual/365",
    frequency=freq,
    multiplier=1.0
)
flo_price = flo.price(df_curve)
flo_sens  = FloatingRateBondSensitivity(flo, df_curve)
print("FRB flottant · Price      :", round(flo_price, 2))
print("             · DV01       :", round(flo_sens.dv01(), 6))
print("             · Duration   :", round(flo_sens.macaulay_duration(), 6))
print("             · Convexity :", round(flo_sens.convexity(), 6))
print()

# === 5) Taux forward discret 1->2 ans ===
start = pricing_date + relativedelta(years=1)
end = pricing_date + relativedelta(years=2)
fwd = ForwardRate(
    pricing_date=pricing_date,
    start_date=start,
    end_date=end,
    convention_days="Actual/365"
)
print(
    f"Forward rate {start.date()} -> {end.date()}: {fwd.value(df_curve) * 100:.2f}%"
)

# === 6) FRA 1->2 ans ===
strike = fwd.value(df_curve)
fra = ForwardRateAgreement(
    notional=notional,
    strike=strike,
    pricing_date=pricing_date,
    start_date=start,
    end_date=end,
    convention_days="Actual/365"
)
print("FRA MtM (payer fixe):", round(fra.mtm(df_curve), 2), "€")

# === 7) Swap payer fixe 5 ans semestriel ===
swap = InterestRateSwap(
    notional=notional,
    fixed_rate=0.035,
    pricing_date=pricing_date,
    maturity_date=maturity_date,
    convention_days="30/360",
    frequency=freq,
    multiplier=1.0,
    margin=0.0,
    forward_curve=fwd_curve
)
swap_mtm   = swap.mtm(df_curve, fwd_curve)
swap_par   = swap.swap_rate(df_curve)
swap_sens  = InterestRateSwapSensitivity(swap, df_curve)
print("Swap     · MtM        :", round(swap_mtm, 2), "€")
print("         · Par Rate   :", f"{swap_par*100:.4f}%")
print("         · DV01       :", round(swap_sens.dv01(), 6))
print("         · Duration   :", round(swap_sens.macaulay_duration(), 6))
print("         · Convexity :", round(swap_sens.convexity(), 6))


options = OptionPortfolio([
    Call(K, maturity_date, exercise="european"),
    Put(K, maturity_date, exercise="european"),
    ]
)
strategies = [
    BearCallSpread(strike_sell=K, strike_buy=K+30, pricing_date=pricing_date, maturity_date=maturity_date),
    BullCallSpread(strike_buy=K, strike_sell=K+30, pricing_date=pricing_date, maturity_date=maturity_date),
    ButterflySpread(strike_low=K, strike_mid=K+30, strike_high=K+60, pricing_date=pricing_date, maturity_date=maturity_date),
    Straddle(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
    Strap(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
    Strip(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
    Strangle(lower_strike=K, upper_strike=K+60, pricing_date=pricing_date, maturity_date=maturity_date),
    Condor(strike1=K, strike2=K+15, strike3=K+50, strike4=K+65, pricing_date=pricing_date, maturity_date=maturity_date),
    PutCallSpread(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
]

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

bs_price = engine.bsm.price()
print(f"Prix estimé      : {np.round(bs_price,4)}")

print("\n====== TRINOMIAL TREE PRICING ======")

engine = TreePortfolio(
    market=market_lvmh,
    option_ptf=options,
    pricing_date=pricing_date,
    n_steps=n_steps
)

price = engine.price()
print(f"Prix estimé (Trinomial Tree) : {np.round(price,4)}")

print(f'\n -------- BLACK SCHOLES PRICING --------')

bse = BSPortfolio(market=market_lvmh, option_ptf=options, pricing_date=pricing_date)
bs_price = bse.price()
print(f"Prix estimé      : {np.round(bs_price,4)}")

print(f'\n -------- VANILLA STRATEGIES PRICING --------')

def price_vanilla_strategy(
    strategy,
    market,
    pricing_date,
    n_paths: int,
    n_steps: int,
    seed: int = None,
    alpha: float = 0.05
) -> tuple[float, float, float]:
    """
    Prix Monte-Carlo d'une stratégie vanille en une passe.
    Retourne (prix, CI_bas, CI_haut).
    """
    # 1) Récupère toutes les jambes et leurs quantités
    legs, qtys = zip(*strategy.get_legs())
    # 2) Pack dans un seul OptionPortfolio
    ptf = OptionPortfolio(list(legs), list(qtys))
    # 3) Instancie l'engine sur TOUT le portefeuille
    engine = MonteCarloEngine(
        market       = market,
        option_ptf   = ptf,
        pricing_date = pricing_date,
        n_paths      = n_paths,
        n_steps      = n_steps,
        seed         = seed
    )
    # 4) Calcule prix et intervalle
    prices = engine.price(type="MC")          # vecteur de prix par option
    ci_low, ci_up = engine.price_confidence_interval(type="MC", alpha=alpha)
    # 5) Somme pondérée
    total_price = float(np.dot(prices, qtys))
    total_low   = float(np.dot(ci_low,  qtys))
    total_up    = float(np.dot(ci_up,   qtys))

    plot_strategy_payoff(strategy)

    return total_price, total_low, total_up

for strat in strategies:
    p, low, up = price_vanilla_strategy(
        strategy     = strat,
        market       = market_lvmh,
        pricing_date = pricing_date,
        n_paths      = n_paths,
        n_steps      = n_steps,
        seed         = seed,
        alpha        = 0.05
    )
    print(f"{strat.name:20s} → Prix: {p:.4f} | CI95%: [{low:.4f}, {up:.4f}]")

