from typing import Callable
from market.day_count_convention import DayCountConvention
from rate.product import (
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    InterestRateSwap
)

class RateProductSensitivity:
    """
    Base pour calculer PV, DV01, duration, convexity
    d’un produit à cash-flows ZC.
    """
    def __init__(self,
                 product,
                 df_curve: Callable[[float], float]):
        self.product  = product
        self.df       = df_curve
        # on reconstruit la liste des ZCB
        self.zc_list  = product.build_cashflows_as_zc()
        dcc = DayCountConvention(product.convention_days)
        # maturités (en années) et montants
        self.times    = [
            dcc.year_fraction(product.pricing_date, zb.maturity_date)
            for zb in self.zc_list
        ]
        self.amounts  = [zb.face_value for zb in self.zc_list]

    def pv(self) -> float:
        """Price actuel (en €)."""
        return sum(amt * self.df(t) for t, amt in zip(self.times, self.amounts))

    def dv01(self) -> float:
        """
        DV01 = ∂PV/∂y · 0.0001 =
        −Σ t_i·CF_i·DF(t_i) × 1e-4
        """
        return -sum(t * amt * self.df(t) for t, amt in zip(self.times, self.amounts)) * 1e-4

    def macaulay_duration(self) -> float:
        """Duration de Macaulay."""
        pv = self.pv()
        weighted = sum(t * amt * self.df(t) for t, amt in zip(self.times, self.amounts))
        return weighted / pv

    def convexity(self) -> float:
        """
        Convexity = Σ t_i^2·CF_i·DF(t_i) × 1e-4
        (sensibilité seconde au taux)
        """
        return sum(t**2 * amt * self.df(t) for t, amt in zip(self.times, self.amounts)) * 1e-4


class ZeroCouponSensitivity(RateProductSensitivity):
    def __init__(self,
                 zcb: ZeroCouponBond,
                 df_curve: Callable[[float], float]):
        super().__init__(zcb, df_curve)


class FixedRateBondSensitivity(RateProductSensitivity):
    def __init__(self,
                 frb: FixedRateBond,
                 df_curve: Callable[[float], float]):
        super().__init__(frb, df_curve)


class FloatingRateBondSensitivity(RateProductSensitivity):
    def __init__(self,
                 flo: FloatingRateBond,
                 df_curve: Callable[[float], float]):
        super().__init__(flo, df_curve)


class InterestRateSwapSensitivity:
    """
    Sensitivités pour un IRS : DV01, duration, convexity sur la MtM.
    On calcule DV01 = DV01_float − DV01_fixed.
    """
    def __init__(self,
                 swap: InterestRateSwap,
                 df_curve: Callable[[float], float]):
        self.swap     = swap
        self.df       = df_curve
        # jambe fixe comme un FixedRateBond fictif
        fixed_bond = FixedRateBond(
            face_value    = swap.notional,
            coupon_rate   = swap.fixed_rate or swap.swap_rate(df_curve),
            pricing_date  = swap.pricing_date,
            maturity_date = swap.maturity_date,
            convention_days = swap.dcc.convention,
            frequency     = swap.frequency
        )
        # jambe flottante idem
        # (on n’inclut pas le notional en fin de maturité car déjà dans forward leg)
        float_bond = FloatingRateBond(
            face_value       = swap.notional,
            margin           = swap.margin,
            pricing_date     = swap.pricing_date,
            maturity_date    = swap.maturity_date,
            forecasted_rates = swap.forecasted_rates or [],
            convention_days  = swap.dcc.convention,
            frequency        = swap.frequency,
            multiplier       = swap.multiplier
        )
        self._sens_fixed = FixedRateBondSensitivity(fixed_bond, df_curve)
        self._sens_float = FloatingRateBondSensitivity(float_bond, df_curve)

    def dv01(self) -> float:
        """DV01 du swap = DV01_float − DV01_fixed."""
        return self._sens_float.dv01() - self._sens_fixed.dv01()

    def macaulay_duration(self) -> float:
        """
        Duration approximative du swap = pondération des durations
        des deux jambes par leurs PV respectifs.
        """
        pv_f   = self._sens_fixed.pv()
        pv_fl  = self._sens_float.pv()
        dur_f  = self._sens_fixed.macaulay_duration()
        dur_fl = self._sens_float.macaulay_duration()
        total  = pv_f + pv_fl
        return (pv_fl * dur_fl - pv_f * dur_f) / total

    def convexity(self) -> float:
        """Convexity swap = convexity_float − convexity_fixed."""
        return self._sens_float.convexity() - self._sens_fixed.convexity()

from functools import singledispatch
from rate.product import (  # adapte le chemin si besoin
    ZeroCouponBond,
    FixedRateBond,
    FloatingRateBond,
    InterestRateSwap
)

# ———— DV01 générique ————
@singledispatch
def dv01(product, df_curve):
    raise NotImplementedError(f"DV01 non supporté pour {type(product)}")

@dv01.register(ZeroCouponBond)
@dv01.register(FixedRateBond)
@dv01.register(FloatingRateBond)
def _dv01_zc(product, df_curve):
    return RateProductSensitivity(product, df_curve).dv01()

@dv01.register(InterestRateSwap)
def _dv01_swap(swap, df_curve):
    return InterestRateSwapSensitivity(swap, df_curve).dv01()


# ———— Duration générique ————
@singledispatch
def duration(product, df_curve):
    raise NotImplementedError(f"Duration non supportée pour {type(product)}")

@duration.register(ZeroCouponBond)
@duration.register(FixedRateBond)
@duration.register(FloatingRateBond)
def _dur_zc(product, df_curve):
    return RateProductSensitivity(product, df_curve).macaulay_duration()

@duration.register(InterestRateSwap)
def _dur_swap(swap, df_curve):
    return InterestRateSwapSensitivity(swap, df_curve).macaulay_duration()


# ———— Convexity générique ————
@singledispatch
def convexity(product, df_curve):
    raise NotImplementedError(f"Convexity non supportée pour {type(product)}")

@convexity.register(ZeroCouponBond)
@convexity.register(FixedRateBond)
@convexity.register(FloatingRateBond)
def _conv_zc(product, df_curve):
    return RateProductSensitivity(product, df_curve).convexity()

@convexity.register(InterestRateSwap)
def _conv_swap(swap, df_curve):
    return InterestRateSwapSensitivity(swap, df_curve).convexity()
