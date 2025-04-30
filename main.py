import argparse
from datetime import datetime, timedelta
import numpy as np

from market.market_factory import create_market
from rate.product import (
    ZeroCouponBond, FixedRateBond, FloatingRateBond,
    ForwardRate, ForwardRateAgreement, InterestRateSwap
)
from risk_metrics.rate_product_sensitivity import dv01, duration, convexity
from pricers.structured_pricer import StructuredPricer
from option.option import (
    OptionPortfolio, Call, Put, DigitalCall, DigitalPut,
    UpAndOutCall, DownAndOutPut, UpAndInCall, DownAndInPut
)
from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreePortfolio
from risk_metrics.options_backtest import Backtest

def parse_args():
    parser = argparse.ArgumentParser(
        description="Outil de pricing et de sensibilité pour produits financiers"
    )
    parser.add_argument(
        "--underlying", "-u", required=True,
        help="Ticker du sous-jacent (doit exister dans config.ini)"
    )
    parser.add_argument(
        "--pricing-date", "-p", default=datetime.today().strftime("%Y-%m-%d"),
        help="Date de valorisation (YYYY-MM-DD)"
    )
    subparsers = parser.add_subparsers(dest="category", required=True)

    # Produits de taux
    sp_rate = subparsers.add_parser("rate", help="Pricing et sensibilités produits de taux")
    sp_rate.add_argument("product", choices=["zcb","fixed","float","fra","swap"])
    sp_rate.add_argument("--face", type=float, default=1000.0, help="Nominal / valeur faciale")
    sp_rate.add_argument("--coupon", type=float, default=0.05, help="Taux coupon (fixed)")
    sp_rate.add_argument("--margin", type=float, default=0.0, help="Margin (float)")
    sp_rate.add_argument("--strike", type=float, default=None, help="Strike (FRA)")
    sp_rate.add_argument("--freq", type=int, default=1, help="Fréquence paiements/an")
    sp_rate.add_argument("--start", help="Date début FRA (YYYY-MM-DD)")
    sp_rate.add_argument("--end", help="Date fin FRA (YYYY-MM-DD)")

    # Options vanilles
    sp_opt = subparsers.add_parser("option", help="Pricing et grecs pour options vanilles")
    sp_opt.add_argument("otype", choices=["call","put"], help="Type d'option")
    sp_opt.add_argument("exercise", choices=["european","american"])
    sp_opt.add_argument("--strike", "-k", type=float, required=True)
    sp_opt.add_argument("--maturity", "-m", required=True, help="Date maturité YYYY-MM-DD")
    sp_opt.add_argument("--backtest", action="store_true", help="Faire backtest VaR")

    # Stratégies structurées
    sp_struct = subparsers.add_parser("structured", help="Pricing produits structurés")
    sp_struct.add_argument("strategy", choices=[
        "ReverseConvertible","TwinWin","BonusCertificate",
        "CappedParticipation","DiscountCertificate","ReverseConvertibleBarrier",
        "SweetAutocall"
    ])
    # on peut ajouter plus de params selon strategy...

    # Stratégies d'investissement optionnelles
    sp_strat = subparsers.add_parser("strategy", help="Pricing stratégies d'options vanilles")
    sp_strat.add_argument("strategy", choices=[
        "BearCallSpread","BullCallSpread","ButterflySpread","Straddle","Strap",
        "Strip","Strangle","Condor","PutCallSpread"
    ])
    # strikes etc. pourraient être passés dynamiquement

    return parser.parse_args()

def main():
    args = parse_args()
    pricing_date = datetime.fromisoformat(args.pricing_date)
    # 1) Création du marché
    market = create_market(
        stock=args.underlying,
        pricing_date=pricing_date,
        vol_source="implied",
        hist_window=252,
        curve_method="svensson",
        curve_kwargs=None,
        dcc="Actual/365"
    )
    # 2) Dispatch selon catégorie
    if args.category == "rate":
        # Calcule la courbe discount et zero
        df = market.discount_factor
        if args.product == "zcb":
            zcb = ZeroCouponBond(args.face, pricing_date, pricing_date + timedelta(days=365))
            price = zcb.price(df)
            print(f"Prix ZCB: {price:.2f}")
            print("DV01:", dv01(zcb, df))
            print("Duration:", duration(zcb, df))
            print("Convexity:", convexity(zcb, df))
        elif args.product == "fixed":
            frb = FixedRateBond(args.face, args.coupon, pricing_date, pricing_date + timedelta(days=365*3), frequency=args.freq)
            price = frb.price(df)
            print(f"Prix Fixed: {price:.2f}")
            print("DV01:", dv01(frb, df))
            print("Duration:", duration(frb, df))
            print("Convexity:", convexity(frb, df))
        # etc. pour float, fra, swap...
        # omitted for brevity

    elif args.category == "option":
        K = args.strike
        T = datetime.fromisoformat(args.maturity)
        opt_cls = Call if args.otype == "call" else Put
        opt = opt_cls(K=K, maturity=T, exercise=args.exercise)
        port = OptionPortfolio([opt])
        engine = MonteCarloEngine(market=market, option_ptf=port, pricing_date=pricing_date, n_paths=10000, n_steps=300, seed=42)
        price = engine.price(type="MC")
        print(f"Prix {args.otype}: {price:.4f}")
        # Grecs via BSM:
        bs = engine.bsm  # BSPortfolio
        print("Delta:", bs.delta())
        print("Gamma:", bs.gamma())
        print("Vega: ", bs.vega())
        print("Theta:", bs.theta())
        print("Rho:", bs.rho())
        if args.backtest:
            bt = Backtest(model=engine, shift_date=pricing_date+timedelta(days=14))
            print("VaR TH:", bt.run("TH", alpha=0.05))
            print("VaR MC:", bt.run("MC", alpha=0.05, nb_simu=100))
            print("VaR CF:", bt.run("CF", alpha=0.05, order=2))

    elif args.category == "structured":
        # Exemple minimal
        strat_cls = globals()[args.strategy]
        prod = strat_cls(pricing_date=pricing_date, maturity_date=pricing_date+timedelta(days=365*3), notional=1000)
        pricer = StructuredPricer(market=market, pricing_date=pricing_date, df_curve=market.discount_factor,
                                   n_paths=10000, n_steps=300, seed=42)
        price = prod.price(pricer)
        print(f"Prix {args.strategy}: {price:.2f}")

    elif args.category == "strategy":
        strat_cls = globals()[args.strategy]
        strat = strat_cls(strike=100, pricing_date=pricing_date, maturity_date=pricing_date+timedelta(days=365))
        # Pricing analogique au bloc option
        # omitted for brevity

if __name__ == "__main__":
    main()

# Exemple d'appel :
# python main.py --underlying LVMH rate fixed --face 1000 --coupon 0.06 --freq 2
