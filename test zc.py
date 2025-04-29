from datetime import datetime
import numpy as np
from dateutil.relativedelta import relativedelta
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent
sys.path.append(str(parent_dir))

# Importez vos classes et utilitaires
from rate.zc_curve import ZCFactory
from market.day_count_convention import DayCountConvention
import sys
from data.management.data_retriever import DataRetriever
from pathlib import Path
from rate.product import (
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    ForwardRate,
    ForwardRateAgreement,
    InterestRateSwap
)
if __name__ == "__main__":
    # === Paramètres généraux ===
    valuation_date = datetime(2023, 4, 25)
    maturity       = datetime(2028, 4, 25)
    notional       = 1_000_000
    face_value     = 1_000
    freq           = 2  # semestriel

    DR = DataRetriever("LVMH")

    rfc = DR.get_risk_free_curve(valuation_date)
    fwc = DR.get_floating_curve(valuation_date)

    # === 1) Calibrage de la courbe ZC (Svensson) ===
    zcf = ZCFactory(rfc, fwc, dcc="Actual/365")
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    df_curve, fwd_curve = zcf.get_discound_and_forward_curves(
        method="svensson",
        initial_guess=sv_guess
    )

    # === 2) Zero Coupon Bond ===
    zcb = ZeroCouponBond(
        face_value=face_value,
        pricing_date=valuation_date,
        maturity_date=maturity,
        convention_days="Actual/365"
    )
    print("ZCB price:", round(zcb.price(df_curve), 2))

    # === 3) Fixed Rate Bond 6% semestriel ===
    frb = FixedRateBond(
        face_value=face_value,
        coupon_rate=0.06,
        pricing_date=valuation_date,
        maturity_date=maturity,
        convention_days="30/360",
        frequency=freq
    )
    print("Fixed Rate Bond price:", round(frb.price(df_curve), 2))

    # === 4) Floating Rate Bond ===
    # Génération des taux forward pour chaque période de paiement
    dates = frb.generate_schedule(valuation_date, maturity, freq)
    dcc   = DayCountConvention("Actual/365")
    t_js  = [dcc.year_fraction(valuation_date, d) for d in dates]
    forwards = [
        fwd_curve(t_js[i-1] if i>0 else 0.0, t_js[i])
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
    print("Floating Rate Bond price:", round(flo.price(df_curve), 2))

    # === 5) Taux forward discret 1->2 ans ===
    start = valuation_date + relativedelta(years=1)
    end   = valuation_date + relativedelta(years=2)
    fwd = ForwardRate(
        pricing_date=valuation_date,
        start_date=start,
        end_date=end,
        convention_days="Actual/365"
    )
    print(
        f"Forward rate {start.date()} -> {end.date()}: {fwd.value(df_curve)*100:.2f}%"
    )

    # === 6) FRA 1->2 ans ===
    strike = fwd.value(df_curve)
    fra = ForwardRateAgreement(
        notional=notional,
        strike=strike,
        pricing_date=valuation_date,
        start_date=start,
        end_date=end,
        convention_days="Actual/365"
    )
    print("FRA MtM (payer fixe):", round(fra.mtm(df_curve), 2), "€")

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
    print("Swap MtM (payer fixe):", round(swap.mtm(df_curve), 2), "€")
    print(f"Swap par rate: {swap.swap_rate(df_curve)*100:.2f}%")
