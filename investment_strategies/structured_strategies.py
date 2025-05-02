from investment_strategies.abstract_strategy import Strategy
from pricers.structured_pricer import StructuredPricer
import numpy as np
from rate.products import ZeroCouponBond
from option.option import Option, Call, Put, DigitalCall, UpAndOutCall, DownAndOutPut
from datetime import datetime
from typing import List


class StructuredProduct(Strategy):
    """
    Un produit structuré définit sa mécanique financière et sait se pricer
    via product.price(pricer).
    """
    def __init__(self,
                 name: str,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__(name, pricing_date, maturity_date, convention_days)
        self.notional = notional

    def get_legs(self) -> List[tuple[ZeroCouponBond | Option, float]]:
        raise NotImplementedError

    def price(self, pricer: StructuredPricer, type: str = "MC") -> float:
        pricer.dcc = self.dcc # override du dcc pour inclure celui du produit
        total = 0.0
        # ZCBs
        invested = self.notional
        for leg, sign in self.get_legs():
            if isinstance(leg, ZeroCouponBond):
                pz = pricer.price_zcb(leg)
                total += sign * pz
                invested -= sign * pz
        # Options vanilles
        opt_legs = [(leg, sign) for leg, sign in self.get_legs()
                    if isinstance(leg, Option)]
        leg_prices = []
        for leg, sign in opt_legs:
            engine = pricer.get_mc_engine(leg)
            p = engine.price(type=type)
            leg_prices.append((leg, sign, p))

        # 2) Somme uniquement les achats (sign > 0)
        total_buy = sum(p for _, sign, p in leg_prices if sign > 0)

        # 3) Répartition de l’invested proportionnellement
        for leg, sign, p in leg_prices:
            if sign > 0:
                qty = invested / total_buy
            else:
                qty = 1.0
            total += sign * qty * p
        return total


class Autocallable(StructuredProduct):
    """
    Autocallable générique avec trois barrières et cash-flows path-dependent.
    """
    def __init__(self,
                 name: str,
                 coupon_barrier:     float,
                 call_barrier:       float,
                 protection_barrier: float,
                 frequency:      str,
                 coupon_rate:   float,
                 pricing_date:   datetime,
                 maturity_date:  datetime,
                 notional:       float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__(name, pricing_date, maturity_date, notional, convention_days)
        self.coupon_barrier = coupon_barrier
        self.call_barrier = call_barrier
        self.protection_barrier = protection_barrier
        self.frequency = frequency
        self.coupon_rate = coupon_rate

    def get_legs(self) -> List[tuple[ZeroCouponBond | Option, float]]:
        # Pas de legs statiques par défaut
        return []

    def price(self, pricer: StructuredPricer) -> float:
        pricer.dcc = self.dcc
        #S, _ = pricer.simulate_underlying(self.maturity_date, self.obs_dates)
        S, times, obs_dates = pricer.simulate_underlying(
                   frequency = self.frequency,
                   dcc = self.dcc
        )
        coupon_rates = np.full(len(obs_dates), self.coupon_rate)

        cashflows, times = pricer.compute_autocall_cashflows(
            S,
            self.coupon_barrier,
            self.call_barrier,
            self.protection_barrier,
            coupon_rates,
            obs_dates,
            self.notional
        )
        # actualisation colonne par colonne
        #zero_rates = np.array([pricer.df_curve(t) for t in times])
        #discounted = cashflows * np.exp(- zero_rates * times)
        discounted = pricer.discount_cf(cashflows=cashflows,times=times)
        return float(discounted.sum(axis=1).mean())


class SweetAutocall(Autocallable):
    """
    Sweet Autocall paramétrable :
       - freq : fréquence de paiement (TRIMESTRIEL, SEMESTRIEL, ANNUEL)
      - coupon_rates : liste des taux annuels correspondants
      - coupon_barrier, call_barrier, protection_barrier : fractions de S₀
    """
    def __init__(self,
                 coupon_rate:         float,
                 frequency:           str,
                 pricing_date:        datetime,
                 maturity_date:       datetime,
                 coupon_barrier:      float = 0.8,
                 call_barrier:        float = 1.1,
                 protection_barrier:  float = 0.8,
                 notional:            float = 1000.0,
                 convention_days:     str = "Actual/365"):


        super().__init__(
            name="Sweet Autocall",
            coupon_barrier=coupon_barrier,
            call_barrier=call_barrier,
            protection_barrier=protection_barrier,
            frequency=frequency,
            coupon_rate=coupon_rate,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            notional=notional,
            convention_days=convention_days
        )


class ReverseConvertible(StructuredProduct):
    def __init__(self,
                 K: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__("Reverse Convertible", pricing_date, maturity_date, notional, convention_days)
        self.K = K

    def get_legs(self):
        return [
            (ZeroCouponBond(face_value=self.notional, pricing_date=self.pricing_date, maturity_date=self.maturity_date), +1.0),
            (Put(self.K, self.maturity_date), -1.0)
        ]


class TwinWin(StructuredProduct):
    def __init__(self,
                 K: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 PDO_barrier: float,
                 CUO_barrier: float,
                 notional: float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__("Twin Win", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.PDO_barrier = PDO_barrier
        self.CUO_barrier = CUO_barrier

    def get_legs(self):
        return [
            (ZeroCouponBond(face_value=self.notional, pricing_date=self.pricing_date, maturity_date=self.maturity_date), +1.0),
            (UpAndOutCall(self.K, self.maturity_date, self.CUO_barrier), +1.0),
            (DownAndOutPut(self.K, self.maturity_date, self.PDO_barrier), +1.0)
        ]


class BonusCertificate(StructuredProduct):
    def __init__(self,
                 K: float,
                 barrier: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__("Bonus Certificate", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.barrier = barrier

    def get_legs(self):
        return [
            (ZeroCouponBond(face_value=self.notional, pricing_date=self.pricing_date, maturity_date=self.maturity_date), +1.0),
            (Call(self.K, self.maturity_date), +1.0),
            (DownAndOutPut(self.K, self.maturity_date, self.barrier), -1.0)
        ]


class CappedParticipationCertificate(StructuredProduct):
    def __init__(self,
                 K: float,
                 cap: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__("Capped Participation Certificate", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.cap = cap

    def get_legs(self):
        return [
            (ZeroCouponBond(face_value=self.notional, pricing_date=self.pricing_date, maturity_date=self.maturity_date), +1.0),
            (Call(self.K, self.maturity_date), +1.0),
            (DigitalCall(self.K, self.maturity_date, "european", payoff=self.cap - self.K), +1.0)
        ]


class DiscountCertificate(StructuredProduct):
    def __init__(self,
                 K: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__("Discount Certificate", pricing_date, maturity_date, notional, convention_days)
        self.K = K

    def get_legs(self):
        return [
            (ZeroCouponBond(face_value=self.notional, pricing_date=self.pricing_date, maturity_date=self.maturity_date), +1.0),
            (Put(self.K, self.maturity_date), -1.0)
        ]


class ReverseConvertibleBarrier(StructuredProduct):
    def __init__(self,
                 K: float,
                 barrier: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__("Reverse Convertible Barrier", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.barrier = barrier

    def get_legs(self):
        return [
            (ZeroCouponBond(face_value=self.notional, pricing_date=self.pricing_date, maturity_date=self.maturity_date), +1.0),
            (Put(self.K, self.maturity_date), -1.0),
            (DownAndOutPut(self.K, self.maturity_date, barrier=self.barrier), +1.0)
        ]

if __name__ == "__main__":
    from market.market_factory import create_market

    # === 1) Définir la date de pricing et la maturité (5 ans) ===
    pricing_date = datetime(2023, 4, 25)
    maturity_date = datetime(2028, 4, 25)

    # === 2) Paramètres pour Svensson ===
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]
    # === 3) Instanciation « tout‐en‐un » du Market LVMH ===
    market_lvmh = create_market(
        stock="LVMH",
        pricing_date=pricing_date,
        vol_source="implied",  # ou "historical"
        hist_window=252,
        curve_method="svensson",  # méthode de calibration
        curve_kwargs={"initial_guess": sv_guess},
        dcc="Actual/Actual",
    )

    # market_lvmh.div_date = datetime(2026,1,1) # à ajouter dans le create_market si demandé par l'user
    # market_lvmh.div_type = "discrete" # à ajouter dans le create_market si demandé par l'user
    # market_lvmh.dividend = 15 # à ajouter dans le create_market si demandé par l'user

    # strike « out‐of‐the‐money » (ici 90% de S0)
    K = market_lvmh.S0 * 0.9

    # barrière “up” à 120% de S0,
    # barrière “down” à 80% de S0
    barrier_up = market_lvmh.S0 * 1.2
    barrier_down = market_lvmh.S0 * 0.8

    # === 2) StructuredPricer ===
    pricer = StructuredPricer(
        market=market_lvmh,
        pricing_date=pricing_date,
        df_curve=market_lvmh.discount,
        maturity_date=maturity_date,
        n_paths=10_000,
        n_steps=300,
        seed=2,
        compute_antithetic=True
    )

    # === 4) Instanciation des produits structurés ===
    products = [
        SweetAutocall(
            coupon_rate=0.1,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            frequency="Annuel",
            coupon_barrier=0.8,  # 80% de S0
            call_barrier=1.1,  # 110% de S0
            protection_barrier=0.8,  # 80% de S0
            notional=1_000.0,
            convention_days="Actual/365"
        ),
        TwinWin(
            K=K,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            PDO_barrier=barrier_down,  # 80% de S0
            CUO_barrier=barrier_up,  # 120% de S0
            notional=1_000.0
        ),
        ReverseConvertible(
            K=K,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            notional=1_000.0
        ),
        BonusCertificate(
            K=K,
            barrier=barrier_down,  # 80% de S0
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            notional=1_000.0
        ),
        CappedParticipationCertificate(
            K=K,
            cap=barrier_up,  # 120% de S0
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            notional=1_000.0
        ),
        DiscountCertificate(
            K=K,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            notional=1_000.0
        ),
        ReverseConvertibleBarrier(
            K=K,
            barrier=barrier_down,
            pricing_date=pricing_date,
            maturity_date=maturity_date,
            notional=1_000.0
        )
    ]

    # === 5) Pricing et affichage ===
    print("\n====== PRICING DES PRODUITS STRUCTURÉS ======\n")
    for prod in products:
        price = prod.price(pricer)
        pct   = price / prod.notional * 100
        print(f"{prod.name:35s} → {pct:6.2f}% du notional")