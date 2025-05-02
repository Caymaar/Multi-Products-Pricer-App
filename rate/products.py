from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Callable, Optional, Tuple
from market.day_count_convention import DayCountConvention


class Bond(ABC):
    """
    Un Bond est défini par ses cash-flows. On le price en rebuildant
    une liste de ZeroCouponBond (un ZCB par cash-flow).
    """

    def __init__(self,
                 face_value:      float,
                 pricing_date:    datetime,
                 maturity_date:   datetime,
                 convention_days: str = "Actual/365"):
        self.face_value     = face_value
        self.pricing_date   = pricing_date
        self.maturity_date  = maturity_date
        self.convention_days= convention_days

    @abstractmethod
    def build_cashflows_as_zc(self):
        """
        Renvoie une liste de ZCB, un par cash-flow.
        """
        pass

    def price(self, df_curve: Callable[[float], float]) -> float:
        """
        Prix du bond en % du nominal.
        """
        zc_bonds = self.build_cashflows_as_zc()
        pv       = sum(zb.price(df_curve) for zb in zc_bonds)
        return pv / self.face_value * 100.0


class ZeroCouponBond(Bond):
    """
    Simple ZCB : un seul cash-flow à maturity_date = face_value.
    """
    def build_cashflows_as_zc(self):
        return [self]

    def price(self, df_curve: Callable[[float], float]) -> float:
        t  = DayCountConvention(self.convention_days) \
                .year_fraction(self.pricing_date, self.maturity_date)
        df = df_curve(t)
        return df * self.face_value


class FixedRateBond(Bond):
    """
    Bond à coupon fixe :
      - chaque période, on paie coupon_rate * face_value * Δ
      - à maturité, on rembourse face_value
    """

    def __init__(self,
                 face_value:      float,
                 coupon_rate:     float,    # en décimal, ex. 0.05 pour 5%
                 pricing_date:    datetime,
                 maturity_date:   datetime,
                 convention_days: str    = "Actual/365",
                 frequency:       str    = "Annuel"):
        super().__init__(face_value, pricing_date, maturity_date, convention_days)
        self.coupon_rate = coupon_rate
        self.frequency   = frequency
        self.dcc         = DayCountConvention(convention_days)

    def build_cashflows_as_zc(self) -> List[ZeroCouponBond]:
        # 1) calendrier
        dates   = self.dcc.schedule(self.pricing_date,
                                     self.maturity_date,
                                     self.frequency)
        # 2) accrual factors
        prev    = self.pricing_date
        taus    = []
        for d in dates:
            taus.append(self.dcc.year_fraction(prev, d))
            prev = d

        # 3) génération des ZCB
        cf_zc: List[ZeroCouponBond] = []
        for tau, pay_date in zip(taus, dates):
            amt = self.face_value * self.coupon_rate * tau
            if pay_date == self.maturity_date:
                amt += self.face_value
            cf_zc.append(
                ZeroCouponBond(
                    face_value     = amt,
                    pricing_date   = self.pricing_date,
                    maturity_date  = pay_date,
                    convention_days= self.convention_days
                )
            )
        return cf_zc


class FloatingRateBond(Bond):
    """
    Bond à coupon variable :
      - chaque période, on paie (multiplier * fwd + margin)*face_value*Δ
      - fwd pris sur E3M si forward_curve fourni, sinon bootstrappé OIS via discount_curve
      - à maturité, on rembourse face_value
    """

    def __init__(self,
                 face_value:      float,
                 margin:          float,    # spread sur la jambe float
                 pricing_date:    datetime,
                 maturity_date:   datetime,
                 convention_days: str                     = "Actual/365",
                 frequency:       str                     = "Annuel",
                 multiplier:      float                   = 1.0,
                 forward_curve:   Callable[[float,float], float] | None = None,
                 discount_curve:  Callable[[float], float]       | None = None):
        super().__init__(face_value, pricing_date, maturity_date, convention_days)
        self.margin         = margin
        self.frequency      = frequency
        self.multiplier     = multiplier
        self.forward_curve  = forward_curve
        self.discount_curve = discount_curve
        self.dcc            = DayCountConvention(convention_days)

        if forward_curve is None and discount_curve is None:
            raise ValueError("Il faut fournir au moins discount_curve si pas de forward_curve.")

    def build_cashflows_as_zc(self) -> List[ZeroCouponBond]:
        # 1) calendrier
        dates    = self.dcc.schedule(self.pricing_date,
                                     self.maturity_date,
                                     self.frequency)
        # 2) accruals
        prev     = self.pricing_date
        taus     = []
        for d in dates:
            taus.append(self.dcc.year_fraction(prev, d))
            prev = d

        # 3) cashflows
        cf_zc: List[ZeroCouponBond] = []
        t_prev   = 0.0
        for tau, pay_date in zip(taus, dates):
            t_i = self.dcc.year_fraction(self.pricing_date, pay_date)

            # forward discret
            if self.forward_curve is not None:
                fwd = self.forward_curve(t_prev, t_i)
            else:
                df_prev = self.discount_curve(t_prev)
                df_i    = self.discount_curve(t_i)
                fwd     = (df_prev / df_i - 1.0) / tau

            amt = (self.multiplier * fwd + self.margin) \
                  * self.face_value * tau

            if pay_date == self.maturity_date:
                amt += self.face_value

            cf_zc.append(
                ZeroCouponBond(face_value      = amt,
                               pricing_date    = self.pricing_date,
                               maturity_date   = pay_date,
                               convention_days = self.convention_days)
            )
            t_prev = t_i

        return cf_zc


class ForwardRate:
    """
    Calcule le taux forward discret entre deux dates > aujourd'hui,
    soit à partir d'une discount curve OIS, soit à partir d'une zero‐coupon
    curve Euribor3M.
    """

    def __init__(self,
                 pricing_date:   datetime,
                 start_date:     datetime,
                 end_date:       datetime,
                 convention_days: str = "Actual/365"):
        assert start_date < end_date, "start_date doit précéder end_date"
        self.pricing_date = pricing_date
        self.start_date   = start_date
        self.end_date     = end_date
        self.dcc          = DayCountConvention(convention_days)

    def _time_fracs(self) -> tuple[float,float]:
        """
        Retourne (t1, t2) en années fractionnées selon la convention.
        """
        t1 = self.dcc.year_fraction(self.pricing_date, self.start_date)
        t2 = self.dcc.year_fraction(self.pricing_date, self.end_date)
        return t1, t2

    def value(self,
              discount_df: Optional[Callable[[float], float]] = None,
              forward_zc = None
              ) -> float:
        """
        Si on fournit discount_df :
            f = (DF(t1)/DF(t2) - 1) / (t2 - t1)
        Sinon si on fournit forward_zc :
            f = forward_zc(t1, t2)
        """
        t1, t2 = self._time_fracs()

        if discount_df is not None and forward_zc is None:
            df1 = discount_df(t1)
            df2 = discount_df(t2)
            # forward discret implicite OIS
            return (df1/df2 - 1) / (t2 - t1)

        if forward_zc is not None and discount_df is None:
            # forward implicite à partir de la zero‐coupon curve Euribor
            return forward_zc(t1, t2)

        raise ValueError(
            "Vous devez fournir soit discount_df, soit forward_zc, mais pas les deux."
        )


class ForwardRateAgreement:
    """
    FRA payeur fixe / receveur float :
      MtM = N · δ · (f_forward − K) · DF(0→t₂)

    On instancie sans strike : on appelle `price()` pour fixer le strike at-par,
    puis `mtm()` pourra être appelé autant de fois que nécessaire avec de
    nouvelles courbes.
    """

    def __init__(self,
                 notional:       float,
                 pricing_date:   datetime,
                 start_date:     datetime,
                 end_date:       datetime,
                 convention_days: str = "Actual/365"):
        self.notional     = notional
        self.pricing_date = pricing_date
        self.start_date   = start_date
        self.end_date     = end_date
        self.dcc          = DayCountConvention(convention_days)
        self.strike       = None  # fixé par price()

    def _time_fracs(self) -> Tuple[float, float, float]:
        t1  = self.dcc.year_fraction(self.pricing_date, self.start_date)
        t2  = self.dcc.year_fraction(self.pricing_date, self.end_date)
        dt   = self.dcc.year_fraction(self.start_date,   self.end_date)
        return t1, t2, dt

    def price(self,
              forward_zc) -> float:
        """
        Pricing initial : calcule et stocke le strike « at par »
        pour lequel MtM₀ = 0.
        """
        t1, t2, _ = self._time_fracs()
        try: # Cas 3M index
            self.strike = forward_zc(t1,t2)
            return self.strike
        except: # Cas O/N index à partir de courbe de discount
            df1 = forward_zc(t1)
            df2 = forward_zc(t2)
            self.strike = (df1/df2 - 1) / (t2 - t1)
            return self.strike

    def mtm(self,
            discount_df: Callable[[float], float],
            forward_zc) -> float:
        """
        Re-valorisation du FRA avec le strike précédemment fixé.
        """
        if self.strike is None:
            raise RuntimeError("Appelez d'abord `price()` pour fixer le strike.")
        t1, t2, dt = self._time_fracs()
        try:  # Cas 3M index
            fwd = forward_zc(t1, t2)
        except:  # Cas O/N index à partir de courbe de discount
            df1 = forward_zc(t1)
            df2 = forward_zc(t2)
            fwd = (df1 / df2 - 1) / (t2 - t1)
        df2 = discount_df(t2)
        return self.notional * dt * (fwd - self.strike) * df2


class InterestRateSwap:
    """
    Vanilla IRS payer fixe / receveur flottant, avec spread et multiplicateur
    sur la jambe flottante, et fréquence unique ('Annuel'/'Semestriel'/'Trimestriel').

    Attributes (après init) :
      - pricing_date, start_date, maturity_date, swap_rate, spread, multiplier, notional
      - dcc : DayCountConvention
      - _t0   : float = year_fraction(pricing_date, start_date)
      - _dates: List[datetime] des dates de paiement (exclu start_date, inclus maturity_date)
      - _taus : List[float] accrual factors τ_i = year_fraction(date_{i-1}, date_i)
      - _times: List[float] times t_i = year_fraction(pricing_date, date_i)
    """

    def __init__(self,
                 notional: float,
                 pricing_date: datetime,
                 start_date: datetime,
                 end_date: datetime,
                 frequency: str,  # 'Annuel' / 'Semestriel' / 'Trimestriel'
                 spread: float = 0.0,
                 multiplier: float = 1.0,
                 convention_days: str = "Actual/365"):
        self.frequency = frequency
        self.notional = notional
        self.pricing_date = pricing_date
        self.start_date = start_date
        self.maturity_date = end_date
        self.spread = spread
        self.multiplier = multiplier
        self.dcc = DayCountConvention(convention_days)
        self.fixed_rate = None

        # time zéro relatif à pricing_date
        self._t0 = self.dcc.year_fraction(self.pricing_date, self.start_date)

        # calendrier commun (start exclu, maturity inclus)
        self._dates: List[datetime] = self.dcc.schedule(
            start_date=self.start_date,
            end_date=self.maturity_date,
            frequency=self.frequency
        )

        # accrual factors τ_i
        prev = self.start_date
        self._taus: List[float] = []
        for d in self._dates:
            tau = self.dcc.year_fraction(prev, d)
            self._taus.append(tau)
            prev = d

        # times t_i (depuis pricing_date) pour actualisation
        self._times: List[float] = [
            self.dcc.year_fraction(self.pricing_date, d)
            for d in self._dates
        ]

    def swap_rate(self,
                 discount_df: Callable[[float], float],
                 forward_zc
                 ) -> float:
        """
        PV_float = ∑ τ_i·DF(t_i)·[M·f_i + s]
        PV_fixed = K * · ∑ τ_i·DF(t_i)
        (avec f_i = forward_zc(t_{i-1}, t_i))
        """
        df_prev      = discount_df(self._t0)
        t_prev       = self._t0
        sum_tau_df   = 0.0
        sum_tau_df_f = 0.0

        for tau, t_i in zip(self._taus, self._times):
            df_i = discount_df(t_i)
            f_i  = forward_zc(t_prev, t_i)
            sum_tau_df   += tau * df_i
            sum_tau_df_f += tau * df_i * f_i
            df_prev = df_i
            t_prev  = t_i

        # 2) PV de la jambe flottante = M·sum_tau_df_f  +  s·sum_tau_df
        pv_float = self.multiplier * sum_tau_df_f \
                   + self.spread   * sum_tau_df

        # 3) on en déduit le par_rate
        self.fixed_rate = pv_float / sum_tau_df
        return self.fixed_rate

    def mtm(self,
            discount_df: Callable[[float], float],
            forward_zc
            ) -> float:
        """
        Valeur de marché du swap = PV(float leg) − PV(fixed leg au swap_rate).
        """
        # --- PV floating leg ---
        df_prev = discount_df(self._t0)
        pv_float = 0.0
        t_prev = self._t0
        for tau, t_i in zip(self._taus, self._times):
            df_i = discount_df(t_i)
            f_i = forward_zc(t_prev, t_i)
            pv_float += tau * df_i * (self.multiplier * f_i + self.spread)
            df_prev = df_i
            t_prev = t_i
        pv_float *= self.notional

        # --- PV fixed leg ---
        if self.fixed_rate is None:
            self.swap_rate(discount_df, forward_zc)
        pv_fixed = self.fixed_rate * sum(
            tau * discount_df(t_i)
            for tau, t_i in zip(self._taus, self._times)
        ) * self.notional

        return pv_float - pv_fixed

if __name__ == "__main__":
    from rate.products import ZeroCouponBond, FixedRateBond, FloatingRateBond, InterestRateSwap
    from dateutil.relativedelta import relativedelta
    from data.management.data_retriever import DataRetriever
    from rate.zc_curve import ZCFactory
    from risk_metrics.rate_product_sensitivity import dv01, duration, convexity

    # === Paramètres généraux ===
    valuation_date = datetime(2023, 4, 25)
    maturity_date = datetime(2028, 4, 25)
    notional = 1_000_000
    face_value = 1_000
    frequency = "Semestriel"

    # === Market data & courbes ===
    DR = DataRetriever("LVMH")
    rfc = DR.get_risk_free_curve(valuation_date)  # OIS-ESTR
    fwc = DR.get_floating_curve(valuation_date)  # Euribor 3M

    zcf = ZCFactory(rfc, fwc, dcc="Actual/365")
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    df_curve = zcf.discount_curve(method="svensson", initial_guess=sv_guess)
    forward_zc = zcf.forward_curve(method="svensson", initial_guess=sv_guess)

    # === Instanciation des produits ===
    zcb = ZeroCouponBond(
        face_value=face_value,
        pricing_date=valuation_date,
        maturity_date=maturity_date,
        convention_days="Actual/365"
    )

    frb = FixedRateBond(
        face_value=face_value,
        coupon_rate=0.06,
        pricing_date=valuation_date,
        maturity_date=maturity_date,
        convention_days="30/360",
        frequency=frequency
    )

    flo = FloatingRateBond(
        face_value=face_value,
        margin=0.002,
        pricing_date=valuation_date,
        maturity_date=maturity_date,
        convention_days="Actual/365",
        frequency=frequency,
        multiplier=1.0,
        forward_curve=forward_zc
    )

    start = valuation_date + relativedelta(years=1)
    end = valuation_date + relativedelta(years=2)
    fra = ForwardRateAgreement(
        notional=notional,
        pricing_date=valuation_date,
        start_date=start,
        end_date=end,
        convention_days="Actual/365"
    )
    fra.price(forward_zc=forward_zc)

    swap = InterestRateSwap(
        notional=notional,
        pricing_date=valuation_date,
        start_date=valuation_date,
        end_date=maturity_date,
        frequency=frequency,
        spread=0.0,
        multiplier=1.0
    )
    swap.swap_rate(discount_df=df_curve, forward_zc=forward_zc)

    # === Prix ===
    print(f"ZCB price:               {zcb.price(df_curve):.2f}")
    print(f"Fixed Rate Bond price:    {frb.price(df_curve):.2f}")
    print(f"Floating Rate Bond price: {flo.price(df_curve):.2f}")
    print(f"FRA fair rate:            {fra.strike * 100:.2f}%")
    print(f"Swap fair par_rate:       {swap.fixed_rate * 100:.4f}%\n")

    # === Sensitivités DV01 ===
    print(f"DV01 ZCB:                 {dv01(zcb, df_curve):.4f}")
    print(f"DV01 Fixed Bond:          {dv01(frb, df_curve):.4f}")
    print(f"DV01 Floating Bond:       {dv01(flo, df_curve):.4f}")
    print(f"DV01 Swap:                {dv01(swap, df_curve, forward_zc):.4f}\n")

    # === Sensitivités Duration ===
    print(f"Duration ZCB:             {duration(zcb, df_curve):.4f}")
    print(f"Duration Fixed Bond:      {duration(frb, df_curve):.4f}")
    print(f"Duration Floating Bond:   {duration(flo, df_curve):.4f}")
    print(f"Duration Swap:            {duration(swap, df_curve, forward_zc):.4f}\n")

    # === Sensitivités Convexity ===
    print(f"Convexity ZCB:            {convexity(zcb, df_curve):.4f}")
    print(f"Convexity Fixed Bond:     {convexity(frb, df_curve):.4f}")
    print(f"Convexity Floating Bond:  {convexity(flo, df_curve):.4f}")
    print(f"Convexity Swap:           {convexity(swap, df_curve, forward_zc):.4f}")