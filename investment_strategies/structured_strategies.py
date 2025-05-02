from investment_strategies.abstract_strategy import Strategy
from pricers.structured_pricer import StructuredPricer
import numpy as np
from rate.product import ZeroCouponBond
from option.option import Option, Call, Put, DigitalCall, UpAndOutCall, DownAndOutPut
from datetime import datetime
from typing import List
from dateutil.relativedelta import relativedelta


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

        # 2) Sommez uniquement les achats (sign > 0)
        total_buy = sum(p for _, sign, p in leg_prices if sign > 0)

        # 3) Répartissez l’invested proportionnellement
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
                 obs_dates:      List[datetime],
                 coupon_rates:   List[float],
                 pricing_date:   datetime,
                 maturity_date:  datetime,
                 notional:       float = 1000.0,
                 convention_days: str = "Actual/365"):
        super().__init__(name, pricing_date, maturity_date, notional, convention_days)
        self.coupon_barrier = coupon_barrier
        self.call_barrier = call_barrier
        self.protection_barrier = protection_barrier
        self.obs_dates = obs_dates
        self.coupon_rates = np.array(coupon_rates)

    def get_legs(self) -> List[tuple[ZeroCouponBond | Option, float]]:
        # Pas de legs statiques par défaut
        return []

    def price(self, pricer: StructuredPricer) -> float:
        pricer.dcc = self.dcc
        #S, _ = pricer.simulate_underlying(self.maturity_date, self.obs_dates)
        S, times = pricer.simulate_underlying(
                   maturity_date = pricer.maturity_date,
                   obs_dates = self.obs_dates,
                   dcc = self.dcc
        )
        cashflows, times = pricer.compute_autocall_cashflows(
            S,
            self.coupon_barrier,
            self.call_barrier,
            self.protection_barrier,
            self.coupon_rates,
            self.obs_dates,
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
      - obs_dates : liste de datetime fournie par l’utilisateur
      - coupon_rates : liste des taux annuels correspondants
      - coupon_barrier, call_barrier, protection_barrier : fractions de S₀
    """
    def __init__(self,
                 freq:                str,
                 coupon_rate:         float,
                 pricing_date:        datetime,
                 maturity_date:       datetime,
                 coupon_barrier:      float = 0.8,
                 call_barrier:        float = 1.1,
                 protection_barrier:  float = 0.8,
                 notional:            float = 1000.0,
                 convention_days:     str = "Actual/365"):
        
        if freq.upper() == "TRIMESTRIEL":
            delta = relativedelta(months=3)
        elif freq.upper() == "SEMESTRIEL":
            delta = relativedelta(months=6)
        elif freq.upper() == "ANNUEL":
            delta = relativedelta(years=1)
        else:
            raise ValueError("Fréquence non supportée: {}".format(freq))

        obs_dates = []
        current_date = pricing_date
        while current_date < maturity_date:
            current_date += delta
            if current_date < maturity_date:
                obs_dates.append(current_date)
            else:
                obs_dates.append(maturity_date)
                break

        coupon_rates = [coupon_rate] * len(obs_dates)

        if len(obs_dates) != len(coupon_rates):
            raise ValueError("obs_dates et coupon_rates doivent avoir la même longueur")
        super().__init__(
            name="Sweet Autocall",
            coupon_barrier=coupon_barrier,
            call_barrier=call_barrier,
            protection_barrier=protection_barrier,
            obs_dates=obs_dates,
            coupon_rates=coupon_rates,
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
