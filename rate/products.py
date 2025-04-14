from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from curve_utils import make_zc_curve

class Bond(ABC):
    def __init__(self, face_value, maturity):
        """
        Classe abstraite pour les obligations.

        :param face_value: Valeur nominale de l'obligation
        :param maturity: Maturité en années
        """
        self.face_value = face_value
        self.maturity = maturity

    @abstractmethod
    def build_cashflows_as_zc(self, zc_curve):
        """Méthode abstraite pour calculer les cashflows comme zero coupon."""
        pass

    def price(self, zc_curve):
        zc_bonds = self.build_cashflows_as_zc(zc_curve)
        return sum(z.price() for z in zc_bonds)

class ZeroCouponBond(Bond):
    def __init__(self, face_value, rate, maturity):
        """
        Obligation zéro coupon.

        :param face_value: Valeur nominale (ex: 1000)
        :param rate: Taux d'intérêt annuel (exprimé en décimal, ex: 0.05 pour 5%)
        :param maturity: Maturité en années
        """
        super().__init__(face_value, maturity)
        self.rate = rate

    def build_cashflows_as_zc(self, zc_curve=None):
        return [self]  # ZC = un seul cashflow

    def price(self, zc_curve=None):
        """Calcul du prix d'un zéro coupon par actualisation de la valeur nominale."""
        return self.face_value * np.exp(-self.rate * self.maturity)

class FixedRateBond(Bond):
    def __init__(self, face_value, coupon_rate, maturity, frequency=1):
        """
        Obligation à taux fixe.

        :param face_value: Valeur nominale de l'obligation (ex: 1000)
        :param coupon_rate: Taux de coupon annuel (ex: 0.06 pour 6%)
        :param maturity: Maturité en années
        :param frequency: Nombre de paiements par an (ex: 2 pour semestriel)
        """
        super().__init__(face_value, maturity)
        self.coupon_rate = coupon_rate
        self.frequency = frequency

    def build_cashflows_as_zc(self, zc_curve):
        dt = 1 / self.frequency
        n = int(self.maturity * self.frequency)
        coupon = self.face_value * self.coupon_rate / self.frequency
        cashflows = []

        for i in range(1, n + 1):
            t = i * dt
            r = zc_curve(t)
            cashflows.append(ZeroCouponBond(coupon, r, t))

        r_T = zc_curve(self.maturity)
        cashflows.append(ZeroCouponBond(self.face_value, r_T, self.maturity))
        return cashflows

class FloatingRateBond(Bond):
    def __init__(self, face_value, margin, maturity, forecasted_rates, frequency=1, multiplier=1.0):
        """
        Obligation à taux variable.

        :param face_value: Valeur nominale de l'obligation
        :param margin: Marge ajoutée au taux de référence (ex: 0.002 pour 0.2%)
        :param maturity: Maturité en années
        :param frequency: Nombre de paiements par an
        :param forecasted_rates: Liste des taux de référence prévisionnels pour chaque période (en décimal).
                                 Si None, on suppose un taux constant de 2%.
        :param multiplier: Niveau du multiplicateur du taux variable de la période
        """
        super().__init__(face_value, maturity)
        self.margin = margin
        self.forecasted_rates = forecasted_rates  # list ou callable
        self.frequency = frequency
        self.multiplier = multiplier

    def build_cashflows_as_zc(self, zc_curve):
        dt = 1 / self.frequency
        n = int(self.maturity * self.frequency)
        cashflows = []

        for i in range(1, n + 1):
            t = i * dt
            r_fwd = self.forecasted_rates[i-1]
            coupon_rate = self.multiplier * r_fwd + self.margin
            coupon = self.face_value * coupon_rate / self.frequency
            r = zc_curve(t)
            cashflows.append(ZeroCouponBond(coupon, r, t))

        r_T = zc_curve(self.maturity)
        cashflows.append(ZeroCouponBond(self.face_value, r_T, self.maturity))
        return cashflows

class ForwardRate:
    def __init__(self, start: float, end: float):
        """
        Représente un taux forward entre deux dates.

        :param start: Date de début (en années)
        :param end: Date de fin (en années)
        """
        assert start < end, "La date de début doit précéder la date de fin"
        self.start = start
        self.end = end

    def value(self, zc_curve):
        """
        Calcule le taux forward continu implicite.
        """
        R_start = zc_curve(self.start)
        R_end = zc_curve(self.end)
        fwd = (R_end * self.end - R_start * self.start) / (self.end - self.start)
        return fwd

class ForwardRateAgreement:
    def __init__(self, notional: float, start: float, end: float, zc_curve):
        """
        Accord de taux à terme (FRA).

        :param notional: Montant notionnel
        :param start: Début de la période (en années)
        :param end: Fin de la période (en années)
        """
        self.notional = notional
        self.start = start
        self.end = end
        self.strike = ForwardRate(self.start,self.end).value(zc_curve)

    def mtm(self, zc_curve):
        """
        Calcule la valeur actuelle du FRA payeur fixe selon la courbe actuelle (0 à l'initialisation).
        """
        fwd = ForwardRate(self.start, self.end).value(zc_curve)
        delta = self.end - self.start
        P_end = np.exp(-zc_curve(self.end) * self.end)
        value = self.notional * delta * (self.strike - fwd) * P_end
        return value

class InterestRateSwap:
    def __init__(self, notional, maturity, frequency, forecasted_rates=None, multiplier=1, margin=0.0, zc_curve=None):
        """
        :param notional: Notional du swap
        :param maturity: Maturité en années
        :param frequency: Fréquence des paiements
        :param forecasted_rates: Liste des taux prévus pour la jambe flottante
        :param margin: Spread appliqué à la jambe flottante
        :param zc_curve: Courbe nécessaire si fixed_rate n’est pas fourni
        """
        self.notional = notional
        self.maturity = maturity
        self.frequency = frequency
        self.forecasted_rates = forecasted_rates
        self.multiplier = multiplier
        self.margin = margin
        self.fixed_rate = self._compute_par_rate(zc_curve)

    def _compute_par_rate(self, zc_curve):
        dt = 1 / self.frequency
        n = int(self.maturity * self.frequency)
        t_i = np.array([i * dt for i in range(1, n + 1)])
        zc = np.exp(-zc_curve(t_i) * t_i)
        numerateur = 1 - zc[-1]
        denominateur = dt * np.sum(zc)
        return numerateur / denominateur

    def mtm(self, zc_curve):
        """
        Calcule la MtM du swap comme différence entre obligation fixe et variable.
        """
        fixed_leg = FixedRateBond(
            face_value=self.notional,
            coupon_rate=self.fixed_rate,
            maturity=self.maturity,
            frequency=self.frequency
        )

        floating_leg = FloatingRateBond(
            face_value=self.notional,
            margin=self.margin,
            maturity=self.maturity,
            forecasted_rates=self.forecasted_rates,
            frequency=self.frequency,
            multiplier=self.multiplier
        )

        return fixed_leg.price(zc_curve) - floating_leg.price(zc_curve)


# Exemple d'utilisation :
if __name__ == "__main__":
    # Chargement des données
    data = pd.read_excel("../data_taux/RateCurve_temp.xlsx")
    maturities = data['Matu'].values
    rates = data['Rate'].values

    # Création de la courbe avec curve_utils (ex: interpolation ici)
    zc_curve = make_zc_curve("interpolation", maturities, rates, kind="cubic")

    # Obligation zéro coupon
    zcb = ZeroCouponBond(face_value=1000, rate=0.05, maturity=5)
    print("Prix du Zero Coupon Bond :", round(zcb.price(), 2))

    # Obligation à taux fixe
    frb = FixedRateBond(face_value=1000, coupon_rate=0.06, maturity=5, frequency=1)
    print("Prix du Fixed Rate Bond :", round(frb.price(zc_curve), 2))

    # Obligation à taux variable
    forecasted_rates = [0.02, 0.021, 0.0205, 0.022, 0.0215]
    forecasted_rates = [0] + forecasted_rates  # Décalage index
    varb = FloatingRateBond(face_value=1000, margin=0.002, maturity=5,
                            forecasted_rates=forecasted_rates, frequency=1)
    print("Prix du Floating Rate Bond :", round(varb.price(zc_curve), 2))

    # Taux forward 1Y -> 2Y
    fwd = ForwardRate(start=1, end=2)
    print("Taux forward 1Y->2Y :", round(fwd.value(zc_curve) * 100, 2), "%")

    # FRA
    fra = ForwardRateAgreement(notional=1_000_000, start=1, end=2, zc_curve=zc_curve)
    print("Prix du FRA (1Y->2Y) :", round(fra.mtm(zc_curve), 2), "€")

    # SWAP
    swap = InterestRateSwap(
        notional=1_000_000,
        maturity=5,
        frequency=1,
        forecasted_rates=forecasted_rates,
        zc_curve=zc_curve
    )

    print(f"Taux fixe swap : {round(swap.fixed_rate * 100, 4)} %")
    print(f"MtM du swap : {round(swap.mtm(zc_curve), 2)} €")
