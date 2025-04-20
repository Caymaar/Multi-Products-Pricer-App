from investment_strategies.strategy import Strategy
from rate.products import ZeroCouponBond
from option.option import Option, Call, Put, DigitalCall, UpAndOutCall, DownAndOutPut
from datetime import datetime

class StructuredProduct(Strategy):
    """
    Un produit structuré définit uniquement sa mécanique financière :
      - notional
      - legs (ZCB et options) et leur sens (+1 achat / -1 vente)
    Le pricer se charge de dimensionner la quantité d’options sur le résiduel.
    """
    def __init__(self, name: str, pricing_date: datetime, maturity_date: datetime, notional: float = 1000.0, convention_days="Actual/365"):
        super().__init__(name, pricing_date, maturity_date, convention_days)
        self.notional = notional # a injecter dans strategy directement et faire la modif pour strat vanilles


    def get_legs(self) -> list[tuple[ZeroCouponBond|Option, float]]:
        """
        Retourne la liste [(instrument, sign)] où :
          - instrument est un ZeroCouponBond ou une Option
          - sign = +1 pour leg long, -1 pour leg short
        """
        pass


class ReverseConvertible(StructuredProduct):
    def __init__(self, K: float, pricing_date: datetime, maturity_date: datetime, notional: float = 1000.0, convention_days="Actual/365"):
        super().__init__("Reverse Convertible", pricing_date, maturity_date, notional, convention_days)
        self.K = K

    def get_legs(self):
        # ZCB protège 100% du notional, Put vendu sur le résiduel
        return [
            (ZeroCouponBond(self.notional, self.ttm), +1.0),
            (Put(self.K, self.maturity_date), -1.0)
        ]


class TwinWin(StructuredProduct):
    def __init__(self, K: float, pricing_date: datetime, maturity_date: datetime, PDO_barrier: float, CUO_barrier: float, notional: float = 1000.0, convention_days="Actual/365"):
        super().__init__("Twin Win", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.PDO_barrier = PDO_barrier
        self.CUO_barrier = CUO_barrier

    def get_legs(self):
        # ZCB + straddle: Call + Put longs
        return [
            (ZeroCouponBond(self.notional, self.ttm), +1.0),
            (UpAndOutCall(self.K, self.maturity_date, self.CUO_barrier), +1.0),
            (DownAndOutPut(self.K, self.maturity_date, self.PDO_barrier), +1.0)
        ]


class BonusCertificate(StructuredProduct):
    def __init__(self, K: float, barrier: float, pricing_date: datetime,  maturity_date: datetime, notional: float = 1000.0, convention_days="Actual/365"):
        super().__init__("Bonus Certificate", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.barrier = barrier


    def get_legs(self):
        # ZCB + Call long + Down-and-Out Put short
        return [
            (ZeroCouponBond(self.notional, self.ttm), +1.0),
            (Call(self.K, self.maturity_date), +1.0),
            (DownAndOutPut(self.K, self.maturity_date, self.barrier), -1.0)
        ]


class LeverageCertificate(StructuredProduct):
    def __init__(self,
                 K: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 leverage: float = 2.0,
                 convention_days="Actual/365"):
        """
        Certificate offrant une exposition à effet de levier :
        - financement partiel via un ZCB de taille notional/leverage
        - achat d'un call multiplié par leverage
        """
        super().__init__("Leverage Certificate", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.leverage = leverage

    def get_legs(self):
        # On protège notional/leverage via ZCB, puis on ajoute leverage calls
        zcb_nominal = self.notional / self.leverage
        return [
            (ZeroCouponBond(face_value=zcb_nominal, maturity=self.ttm), +1.0),
            (Call(self.K, self.maturity_date), self.leverage)
        ]


class CappedParticipationCertificate(StructuredProduct):
    def __init__(self,
                 K: float,
                 cap: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days="Actual/365"):
        """
        Participation limitée (cap) :
        - ZCB protégeant 100% du notional
        - Call vanilla pour capter la hausse jusqu’à cap
        - DigitalCall pour abonder jusqu’à cap-K
        """
        super().__init__("Capped Participation Certificate", pricing_date, maturity_date, notional, convention_days)
        self.K = K
        self.cap = cap

    def get_legs(self):
        # On protège 100% du notional
        # Puis on expose un call + un digital pour capper la participation
        return [
            (ZeroCouponBond(face_value=self.notional, maturity=self.ttm), +1.0),
            (Call(self.K, self.maturity_date), +1.0),
            (DigitalCall(self.K, self.maturity_date, "european", payoff=self.cap - self.K), +1.0)
        ]


class DiscountCertificate(StructuredProduct):
    def __init__(self,
                 K: float,
                 pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days="Actual/365"):
        """
        Discount Certificate : versement de min(S_T, K)
        → ZCB (face_value = notional) + vente de Put
        """
        super().__init__("Discount Certificate", pricing_date, maturity_date, notional, convention_days)
        self.K = K

    def get_legs(self):
        # ZCB protègera notional, on vend 1 put unitaire
        return [
            (ZeroCouponBond(face_value=self.notional, maturity=self.ttm), +1.0),
            (Put(self.K, self.maturity_date), -1.0)
        ]


class ReverseConvertibleBarrier(StructuredProduct):
    def __init__(self,
                 K: float,
                 barrier: float,
                    pricing_date: datetime,
                 maturity_date: datetime,
                 notional: float = 1000.0,
                 convention_days="Actual/365"):
        """
        Reverse Convertible KO :
        - ZCB pour notional
        - short Put
        - Down-and-Out Put pour limiter la perte si barrière touchée
        """
        super().__init__("Reverse Convertible Barrier", pricing_date,maturity_date, notional, convention_days)
        self.K = K
        self.barrier  = barrier

    def get_legs(self):
        return [
            (ZeroCouponBond(face_value=self.notional, maturity=self.ttm), +1.0),
            (Put(self.K, self.maturity_date), -1.0),
            (DownAndOutPut(self.K, self.maturity_date, barrier=self.barrier), +1.0)
        ]
