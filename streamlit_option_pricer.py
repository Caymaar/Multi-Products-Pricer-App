# streamlit_pricer_all_categories.py
import streamlit as st
from datetime import datetime, time as _time
from datetime import timedelta
import inspect
from investment_strategies.structured_strategies import SweetAutocall

from market.market_factory import create_market
from pricers.structured_pricer import StructuredPricer
from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreePortfolio
from pricers.bs_pricer import BSPortfolio
from rate.product import (
    ZeroCouponBond, FixedRateBond, FloatingRateBond,
    ForwardRateAgreement, InterestRateSwap
)
from risk_metrics.greeks import GreeksCalculator
from option.option import OptionPortfolio
from market.day_count_convention import DayCountConvention

from app import Category, COMMON_PARAMS, SPECIFIC_PARAMS, plot_strategy_payoff
import pandas as pd

def get_init_parameters(cls_or_fn):
    if inspect.isclass(cls_or_fn):
        sig = inspect.signature(cls_or_fn.__init__)
    else:
        sig = inspect.signature(cls_or_fn)
    return [p for p in sig.parameters if p != 'self']

# Sidebar
st.sidebar.header("🛠️ Marché & MC/Tree/BS")
underlying       = st.sidebar.text_input("Ticker", "LVMH")
raw_pricing_date = st.sidebar.date_input("Date de valorisation", datetime.today().date(), key="pricing_date")
pricing_date     = datetime.combine(raw_pricing_date, _time.min)
vol_source       = st.sidebar.selectbox("Vol source", ["implied","historical"], key="vol_source")
hist_window      = st.sidebar.number_input("Hist window", 252, key="hist_window")
curve_method     = st.sidebar.selectbox("Courbe", ["interpolation","nelson-siegel","svensson"], key="curve_method")
dcc              = st.sidebar.selectbox("Day count", ["Actual/360","Actual/365","30/360"], key="dcc")

n_paths    = st.sidebar.number_input("MC chemins", 10000, step=1000, key="n_paths")
n_steps    = st.sidebar.number_input("MC steps", 300, step=10, key="n_steps")
seed       = st.sidebar.number_input("Seed", 42, key="seed")
tree_steps = st.sidebar.number_input("Tree steps", 100, step=10, key="tree_steps")
bs_vol     = st.sidebar.number_input("BS vol", 0.2, key="bs_vol")

market = create_market(
    stock=underlying, pricing_date=pricing_date,
    vol_source=vol_source, hist_window=hist_window,
    curve_method=curve_method, curve_kwargs=None, dcc=dcc
)

st.title("💡 Pricer multi-catégories")

tabs = st.tabs([c.name for c in Category])
for tab, category in zip(tabs, Category):
    with tab:
        st.header(category.name)

        # select product
        ProdEnum = category.value
        prod = st.selectbox(
            f"{category.name} →",
            list(ProdEnum),
            format_func=lambda p: p.name,
            key=f"select_{category.name}"
        )

        # build widget list
        common   = COMMON_PARAMS[category]
        specific = SPECIFIC_PARAMS[category].get(prod.name, [])
        specs    = common + specific

        vals = {}
        for s in specs:
            w = getattr(st, s["widget"])
            kw = s["kwargs"].copy()
            if "value_func" in kw:
                kw["value"] = kw.pop("value_func")(pricing_date)
            key = f"{category.name}-{prod.name}-{s['name']}"
            vals[s["name"]] = w(s["label"], key=key, **kw)

        if prod.value is SweetAutocall:
            vals["obs_dates"] = st.date_input(
                "Observation Dates",
                value=[pricing_date + timedelta(days=365*i) for i in (1,2,3,4,5)],
                key="obs_dates",
                help="Sélectionnez une ou plusieurs dates d’observation"
            )

            raw = st.text_input(
                "Coupon Rates (séparés par des virgules)",
                value="0.05,0.05,0.05,0.05,0.05",
                key="coupon_rates",
                help="Entrez les taux annuels séparés par des virgules, p.ex. 0.05,0.05,…"
            )
            # Parsing en liste de floats
            try:
                vals["coupon_rates"] = [float(x.strip()) for x in raw.split(",") if x.strip()!='']
            except ValueError:
                st.error("Format des coupon rates invalide ! Veuillez n’utiliser que des nombres séparés par des virgules.")
                vals["coupon_rates"] = []

        # pricing method selector
        if category is Category.STRUCTURED:
            method = "Structured"
        elif category in (Category.OPTION, Category.STRATEGY):
            method = st.radio("Méthode", ["MC","Tree","BS"], horizontal=True, key=f"mode_{category.name}")
        else:
            method = st.radio("Action", ["price","mtm"], horizontal=True, key=f"mode_{category.name}")

        if st.button(f"▶️ Pricer {prod.name}", key=f"btn_{category.name}"):

            for key, value in vals.items():
                if isinstance(value, (int, float)):
                    if "strike" in key.lower() or "barrier" in key.lower() or key == "K":
                        vals[key] = value * market.S0 /100
            # insert pricing_date if needed
            if "pricing_date" in get_init_parameters(prod.value):
                vals["pricing_date"] = pricing_date

            for k, v in vals.items():
                if isinstance(v, datetime):
                    vals[k] = datetime(v.year, v.month, v.day)


            # dispatch
            if category is Category.STRUCTURED:
                inst = prod.value(**vals)
                pr = StructuredPricer(
                    market=market, pricing_date=pricing_date,
                    df_curve=market.discount, maturity_date=vals["maturity_date"], 
                    n_paths=n_paths, n_steps=n_steps, seed=seed, compute_antithetic=True
                )
                price = inst.price(pr)

            elif category is Category.OPTION:
                inst = prod.value(**vals)
                ptf  = OptionPortfolio([inst])
                if method == "MC":
                    eng = MonteCarloEngine(market, ptf, pricing_date, n_paths, n_steps, seed)
                    price = eng.price(type="MC")
                elif method == "Tree":
                    eng = TreePortfolio(market, ptf, pricing_date, tree_steps)
                    price = eng.price()
                else:
                    eng = MonteCarloEngine(market, ptf, pricing_date, n_paths, n_steps, seed)
                    price = eng.bsm.price()
                
                gc = GreeksCalculator(eng)
                greeks = gc.all_greeks()
                df_results = pd.DataFrame([greeks], columns=["delta", "gamma", "vega", "thtea", "rho", "speed"])
                df_results = df_results.T.rename(columns={0: "value"}).T
                

            elif category is Category.STRATEGY:
                strat = prod.value(**vals)
                legs, qtys = zip(*strat.get_legs())
                ptf = OptionPortfolio(list(legs), list(qtys))
                if method == "MC":
                    eng = MonteCarloEngine(market, ptf, pricing_date, n_paths, n_steps, seed)
                    prices = eng.price(type="MC")
                    price = sum(p*q for p,q in zip(prices, qtys))
                elif method == "Tree":
                    eng = TreePortfolio(market, ptf, pricing_date, tree_steps)
                    prices = eng.price()
                    price = sum(p*q for p,q in zip(prices, qtys))
                else:
                    eng = MonteCarloEngine(market, ptf, pricing_date, n_paths, n_steps, seed)
                    prices = eng.bsm.price()
                    price = sum(p*q for p,q in zip(prices, qtys))

                st.pyplot(plot_strategy_payoff(strat))
                gc = GreeksCalculator(eng)
                greeks = gc.all_greeks()
                df_results = pd.DataFrame([greeks], columns=["delta", "gamma", "vega", "thtea", "rho", "speed"])
                df_results = df_results.T.rename(columns={0: "value"}).T

            else:  # RATE
                inst = prod.value(**vals)
                if method == "price":
                    price = inst.price(market.discount)
                else:
                    price = inst.mtm(market.discount)

            st.success(f"Prix {prod.name} → {price:.4f}")
            # Si df_results est défini, l'afficher
            if 'df_results' in locals():
                st.dataframe(df_results)