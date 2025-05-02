import streamlit as st
from datetime import datetime, time as _time, date
import inspect

from market.market_factory import create_market
from pricers.structured_pricer import StructuredPricer
from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreePortfolio

from investment_strategies.vanilla_strategies import plot_strategy_payoff
from rate.products import (
    ZeroCouponBond, FixedRateBond, FloatingRateBond,
    ForwardRateAgreement, InterestRateSwap
)
from risk_metrics.greeks import GreeksCalculator
from option.option import OptionPortfolio


from app import Category, COMMON_PARAMS, SPECIFIC_PARAMS, Sensitivity
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
raw_pricing_date = st.sidebar.date_input("Date de valorisation", date(2023, 1, 1), key="pricing_date")
pricing_date     = datetime.combine(raw_pricing_date, _time.min)
vol_source       = st.sidebar.selectbox("Vol source", ["implied","historical"], key="vol_source")
hist_window      = st.sidebar.number_input("Hist window", 252, key="hist_window")
curve_method     = st.sidebar.selectbox("Courbe", ["interpolation","nelson-siegel","svensson"], key="curve_method")
dcc              = st.sidebar.selectbox("Day count", ["Actual/360","Actual/365","30/360", "Actual/Actual"], key="dcc")

n_paths    = st.sidebar.number_input("MC chemins", 10000, step=1000, key="n_paths")
n_steps    = st.sidebar.number_input("MC steps", 300, step=10, key="n_steps")
seed       = st.sidebar.number_input("Seed", 42, key="seed")
tree_steps = st.sidebar.number_input("Tree steps", 100, step=10, key="tree_steps")

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

        # pricing method selector
        if category is Category.STRUCTURED:
            method = "Structured"
        elif category in (Category.OPTION, Category.STRATEGY):
            method = st.radio("Méthode", ["MC","Tree","BS"], horizontal=True, key=f"mode_{category.name}")

        if st.button(f"▶️ Pricer {prod.name}", key=f"btn_{category.name}"):

            for key, value in vals.items():
                if isinstance(value, (int, float)):
                    if "strike" in key.lower() or "barrier" in key.lower() or key == "K":
                        vals[key] = value * market.S0 /100
                    if "coupon_rate" in key.lower() or "call_barrier" in key.lower() or "protection_barrier" in key.lower() or "coupon_barrier" in key.lower():
                        vals[key] = value / 100
            if "pricing_date" in get_init_parameters(prod.value):
                vals["pricing_date"] = pricing_date

            if "convention_days" in get_init_parameters(prod.value):
                vals["convention_days"] = dcc


            for k, v in vals.items():
                if isinstance(v, datetime):
                    vals[k] = datetime(v.year, v.month, v.day)
                elif isinstance(v, date):
                    vals[k] = datetime(v.year, v.month, v.day)


            # dispatch
            if category is Category.STRUCTURED:
                inst = prod.value(**vals)
                pr = StructuredPricer(
                    market=market, pricing_date=pricing_date,
                    df_curve=market.discount, maturity_date=vals["maturity_date"], 
                    n_paths=n_paths, n_steps=n_steps, seed=seed, compute_antithetic=True
                )
                price = inst.price(pr) / inst.notional * 100

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
                df_results = pd.DataFrame([greeks], columns=["delta", "gamma", "vega", "theta", "rho", "speed"])
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
                df_results = pd.DataFrame([greeks], columns=["delta", "gamma", "vega", "theta", "rho", "speed"])
                df_results = df_results.T.rename(columns={0: "value"}).T

            elif category is Category.RATE:

                if prod.value in [FloatingRateBond]:
                    vals["forward_curve"] = market.forward

                if prod.value in [ZeroCouponBond, FixedRateBond, FloatingRateBond]:
                    inst = prod.value(**vals)
                    price = inst.price(df_curve=market.discount) / inst.face_value * 100 if prod.value is ZeroCouponBond else inst.price(df_curve=market.discount)
                elif prod.value is InterestRateSwap:
                    inst = prod.value(**vals)
                    swap_rate = inst.swap_rate(discount_df=market.discount, forward_zc=market.forward)
                    price = inst.mtm(discount_df=market.discount,forward_zc=market.forward)
                    st.success(f"Swap Rate → {swap_rate:.4%}")
                    
                else:
                    init_params_fwd = get_init_parameters(ForwardRateAgreement)
                    fwd_vals = {k: vals[k] for k in init_params_fwd if k in vals}
                    
                    fra = ForwardRateAgreement(**fwd_vals)
                    if vals.get("reference_rate").lower() == "3m":
                        rate = fra.price(market.forward)
                        st.success(f"Forward Rate Agreement → {rate:.4%}")
                        price = fra.mtm(market.discount, market.forward)
                    else:
                        rate = fra.price(market.discount)
                        st.success(f"Forward Rate Agreement → {rate:.4%}")
                        price = fra.mtm(market.discount, market.discount)

            if prod.name in Sensitivity.__members__:
                sens_cls = Sensitivity[prod.name].value
                if prod.name == "InterestRateSwap":
                    # On doit passer le forward_zc pour le calcul de la sensibilité
                    forward_zc = market.forward
                    sens = sens_cls(inst, market.discount, forward_zc)
                else:
                    # On passe le discount curve pour les autres produits
                    sens = sens_cls(inst, market.discount)
                sensitivity_data = {
                    "Metric": ["DV01", "Duration", "Convexity"],
                    "Value": [
                        round(sens.dv01(), 6),
                        round(sens.macaulay_duration(), 6),
                        round(sens.convexity(), 6)
                    ]
                }
                df_results = pd.DataFrame(sensitivity_data)
                df_results.set_index("Metric", inplace=True)

            if prod.name in ["ForwardRateAgreement", "InterestRateSwap"]:
                st.success(f"MtM {prod.name} → {price:.4f}")
            else:
                st.success(f"Prix {prod.name} → {price:.4f}")
            # Si df_results est défini, l'afficher
            if 'df_results' in locals():
                st.dataframe(df_results)