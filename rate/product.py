import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from dateutil.relativedelta import relativedelta
from market.day_count_convention import DayCountConvention


# --- Mixin pour le planning et les accruals ---
class ScheduleMixin:
    def generate_schedule(self,
                          pricing_date: datetime,
                          maturity_date: datetime,
                          frequency: int
                         ) -> list[datetime]:
        """
        Retourne la liste des dates de paiement entre pricing_date (exclu)
        et maturity_date (inclus), à fréquence donnée (paiements/an).
        """
        if frequency < 1:
            raise ValueError("frequency doit être ≥ 1")
        dt_months = 12 // frequency
        dates = []
        cur = pricing_date
        while True:
            nxt = cur + relativedelta(months=dt_months)
            if nxt >= maturity_date:
                break
            dates.append(nxt)
            cur = nxt
        dates.append(maturity_date)
        return dates

    def compute_accruals(self,
                        dates: list[datetime],
                        pricing_date: datetime,
                        convention_days: str
                       ) -> np.ndarray:
        """
        Pour une liste de dates (dates[0] = 1er paiement, …), retourne
        les accruals (fractions d'année) entre pricing_date→dates[0], dates[0]→dates[1], ….
        """
        dcc = DayCountConvention(convention_days)
        all_dates = [pricing_date] + dates
        accruals = []
        for i in range(len(dates)):
            tau = dcc.year_fraction(all_dates[i], all_dates[i+1])
            accruals.append(tau)
        return np.array(accruals)  # shape (n_periods,)


# --- Classe de base pour tous les bonds ---
class Bond(ABC, ScheduleMixin):
    def __init__(self,
                 face_value: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 convention_days: str = "Actual/365"):
        self.face_value    = face_value
        self.pricing_date  = pricing_date
        self.maturity_date = maturity_date
        self.convention_days = convention_days

    @abstractmethod
    def build_cashflows_as_zc(self) -> list["ZeroCouponBond"]:
        """
        Renvoie une liste de ZeroCouponBond (un par cash-flow)
        dont la date de maturité est la date de paiement, et le montant
        le flux.
        """
        pass

    def price(self, df_curve) -> float:
        zc_bonds = self.build_cashflows_as_zc()
        return sum(zb.price(df_curve) for zb in zc_bonds) / self.face_value * 100.0
    



# --- Zéro Coupon simple ---
class ZeroCouponBond(Bond):
    def __init__(self,
                 face_value: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 convention_days: str = "Actual/365"):
        super().__init__(face_value, pricing_date, maturity_date, convention_days)

    def build_cashflows_as_zc(self) -> list["ZeroCouponBond"]:
        return [self]

    def price(self, df_curve) -> float:
        t = DayCountConvention(self.convention_days)\
            .year_fraction(self.pricing_date, self.maturity_date)
        df = df_curve(t)
        return df * self.face_value
        

# --- Bond à taux fixe ---
class FixedRateBond(Bond):
    def __init__(self,
                 face_value: float,
                 coupon_rate: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 convention_days: str = "Actual/365",
                 frequency: int = 1):
        super().__init__(face_value, pricing_date, maturity_date, convention_days)
        self.coupon_rate = coupon_rate
        self.frequency   = frequency

    def build_cashflows_as_zc(self) -> list[ZeroCouponBond]:
        # 1) calendrier
        dates = self.generate_schedule(self.pricing_date,
                                       self.maturity_date,
                                       self.frequency)
        # 2) accruals
        accruals = self.compute_accruals(dates,
                                         self.pricing_date,
                                         self.convention_days)
        # 3) montants et ZCB
        cashflows = []
        for i, pay_date in enumerate(dates):
            amount = float(self.face_value * self.coupon_rate * accruals[i])
            if pay_date == self.maturity_date:
                amount += self.face_value
            cashflows.append(
                ZeroCouponBond(amount,
                               self.pricing_date,
                               pay_date,
                               convention_days=self.convention_days)
            )
        return cashflows


# --- Bond à taux variable ---
class FloatingRateBond(Bond):
    def __init__(self,
                 face_value: float,
                 margin: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 forecasted_rates: list[float],
                 convention_days: str = "Actual/365",
                 frequency: int = 1,
                 multiplier: float = 1.0):
        super().__init__(face_value, pricing_date, maturity_date, convention_days)
        self.margin = margin
        self.forecasted_rates = forecasted_rates or []
        self.frequency = frequency
        self.multiplier = multiplier

    def build_cashflows_as_zc(self) -> list[ZeroCouponBond]:
        dates    = self.generate_schedule(self.pricing_date,
                                          self.maturity_date,
                                          self.frequency)
        accruals = self.compute_accruals(dates,
                                         self.pricing_date,
                                         self.convention_days)
        cashflows = []
        for i, pay_date in enumerate(dates):
            idx_rate = min(i, len(self.forecasted_rates)-1)
            fwd = self.forecasted_rates[idx_rate]
            amount = float(self.face_value * (self.multiplier * fwd + self.margin) * accruals[i])
            if pay_date == self.maturity_date:
                amount += self.face_value
            cashflows.append(
                ZeroCouponBond(amount,
                               self.pricing_date,
                               pay_date,
                               convention_days=self.convention_days)
            )
        return cashflows

class ForwardRate:
    """
    Représente le taux forward (discret) entre deux dates > aujourd'hui,
    valorisé aujourd'hui.
    """
    def __init__(self,
                 pricing_date: datetime,
                 start_date:   datetime,
                 end_date:     datetime,
                 convention_days: str = "Actual/365"):
        assert start_date < end_date, "start_date doit précéder end_date"
        self.pricing_date = pricing_date
        self.start_date = start_date
        self.end_date = end_date
        self.dcc = DayCountConvention(convention_days)

    def value(self, df_curve) -> float:
        """
        Taux forward discret implicite sur [t1, t2] :
           f = (DF(0→t1)/DF(0→t2) − 1) / (t2 − t1)
        avec DF(0→ti)=exp(−r(0→ti)*ti).
        """
        t1 = self.dcc.year_fraction(self.pricing_date, self.start_date)
        t2 = self.dcc.year_fraction(self.pricing_date, self.end_date)

        df1 = df_curve(t1)
        df2 = df_curve(t2)

        return (-np.log(df2) + np.log(df1)) / (t2 - t1)

class ForwardRateAgreement:
    """
    Valorisation d’un FRA payeur fixe (et receveur floating) :
      MtM₀ = N · δ · (f_forward − K) · DF(0→t₂)
    où δ = t₂ − t₁, K le strike du FRA, f_forward le taux forward continu implicite.
    """
    def __init__(self,
                 notional:       float,
                 strike:         float,
                 pricing_date:   datetime,
                 start_date:     datetime,
                 end_date:       datetime,
                 convention_days: str = "Actual/365"):
        self.notional = notional
        self.strike = strike
        self.pricing_date = pricing_date
        self.start_date = start_date
        self.end_date = end_date
        self.convention_days = convention_days
        self.dcc = DayCountConvention(convention_days)

    def fair_rate(self, df_curve) -> float:
        """
        Renvoie le taux forward continu implicite f sur [t1, t2].
        """
        frwd = ForwardRate(
            pricing_date = self.pricing_date,
            start_date = self.start_date,
            end_date = self.end_date,
            convention_days = self.convention_days
        )
        return frwd.value(df_curve)

    def mtm(self, df_curve) -> float:
        """
        Calcule la MtM du FRA aujourd'hui avec le strike K fixé.
        Utilise fair_rate() pour récupérer f_forward.
        """
        # 1) calcul des t1, t2 et de delta
        t1 = self.dcc.year_fraction(self.pricing_date, self.start_date)
        t2 = self.dcc.year_fraction(self.pricing_date, self.end_date)
        dt = t2 - t1

        # 2) taux forward implicite
        fwd = self.fair_rate(df_curve)

        # 3) factor d'actualisation continue jusque t2
        DF2  = df_curve(t2)


        # 4) MtM
        return self.notional * dt * (self.strike - fwd) * DF2


# --- Interest Rate Swap ---
class InterestRateSwap(ScheduleMixin):
    """
    Swap payer-fixed vs receive-float:
     • swap_rate(df_curve) → par rate
     • mtm(df_curve)       → MtM aujourd’hui au fixed_rate choisi
    """
    def __init__(self,
                 notional:       float,
                 fixed_rate:     float | None,
                 pricing_date:   datetime,
                 maturity_date:  datetime,
                 convention_days: str,
                 frequency:      int = 1,
                 multiplier:     float = 1.0,
                 margin:         float = 0.0,
                 forecasted_rates:list[float]|None = None):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.pricing_date = pricing_date
        self.maturity_date = maturity_date
        self.frequency  = frequency
        self.multiplier = multiplier
        self.margin = margin
        self.forecasted_rates= forecasted_rates
        self.dcc = DayCountConvention(convention_days)

    def _annuity(self, df_curve) -> float:
        dates = self.generate_schedule(self.pricing_date,
                                          self.maturity_date,
                                          self.frequency)
        accruals = self.compute_accruals(dates,
                                         self.pricing_date,
                                         self.dcc.convention)
        dfs = np.array([
            df_curve(self.dcc.year_fraction(self.pricing_date, d))
            for d in dates
        ])
        return float(np.sum(accruals * dfs))

    def _pv_float_leg(self, df_curve) -> float:
        """
        PV de la jambe flottante = Σ_j N·(multiplier·L_j + margin)·Δ_j·DF(0→t_j)
        où L_j est soit forecasted_rates[j], soit calculé par :
          L_j = (DF(t_{j-1})/DF(t_j) − 1) / Δ_j
        """
        dates = self.generate_schedule(self.pricing_date,
                                          self.maturity_date,
                                          self.frequency)
        accruals = self.compute_accruals(dates,
                                         self.pricing_date,
                                         self.dcc.convention)

        # On calcule les factors d'actualisation
        t_js = np.array([self.dcc.year_fraction(self.pricing_date, d)
                         for d in dates])
        dfs = np.vectorize(df_curve)(t_js)

        # 1) forward implicites si pas de forecasted_rates
        if self.forecasted_rates is None:
            # on a besoin aussi des DF(t_{j-1})
            t_jm1 = np.concatenate([[0.0], t_js[:-1]])
            df_jm1 = np.vectorize(df_curve)(t_jm1)
            forwards = (df_jm1/dfs - 1) / accruals
        else:
            forwards = np.array(self.forecasted_rates)

        # 2) montants de chaque paiement
        payments = (self.multiplier * forwards + self.margin) \
                   * self.notional * accruals

        # 3) PV = somme des payments actualisés
        return float(np.sum(payments * dfs))

    def swap_rate(self, df_curve) -> float:
        """
        Par rate pur = (1-DF(T)) / Σ Δ_j·DF(t_j)
        """
        T = self.dcc.year_fraction(self.pricing_date, self.maturity_date)
        DFt = df_curve(T)
        A = self._annuity(df_curve)
        return (1 - DFt) / A

    def mtm(self, df_curve) -> float:
        """
        MtM = PV_float_leg(total) – PV_fixed_leg
        si fixed_rate=None, on prend swap_rate ⇒ MtM≈0
        """
        R = self.fixed_rate if self.fixed_rate is not None else self.swap_rate(df_curve)
        A = self._annuity(df_curve)
        pv_fixed = self.notional * R * A
        pv_float = self._pv_float_leg(df_curve)
        return float(pv_float - pv_fixed)

# Usage exemple

if __name__ == "__main__":

    from data.management.data_retriever import DataRetriever
    from rate.zc_curve import ZCFactory

    # === Paramètres généraux ===
    valuation_date = datetime(2023, 4, 25)
    maturity = datetime(2028, 4, 25)
    notional = 1_000_000
    face_value = 1_000
    freq = 2  # semestriel

    DR = DataRetriever("LVMH")

    rfc = DR.get_risk_free_curve(valuation_date)
    fwc = DR.get_floating_curve(valuation_date)

    # === 1) Calibrage de la courbe ZC (Svensson) ===
    zcf = ZCFactory(rfc, fwc, dcc="Actual/365")
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]

    df_curve = zcf.discount_curve(
        method="svensson",
        initial_guess=sv_guess
    )

    fwd_curve = zcf.forward_curve(
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
    dcc = DayCountConvention("Actual/365")
    t_js = [dcc.year_fraction(valuation_date, d) for d in dates]
    forwards = [
        fwd_curve(t_js[i - 1] if i > 0 else 0.0, t_js[i])
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
    end = valuation_date + relativedelta(years=2)
    fwd = ForwardRate(
        pricing_date=valuation_date,
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
    print(f"Swap par rate: {swap.swap_rate(df_curve) * 100:.2f}%")
