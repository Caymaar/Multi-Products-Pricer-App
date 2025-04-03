from abc import ABC, abstractmethod

# ---------------- Base Bond Class ----------------


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
    def price(self):
        """Méthode abstraite pour calculer le prix de l'obligation."""
        pass


class ZeroCouponBond(Bond):
    def __init__(self, face_value, yield_rate, maturity):
        """
        Obligation zéro coupon.

        :param face_value: Valeur nominale (ex: 1000)
        :param yield_rate: Taux d'intérêt annuel (exprimé en décimal, ex: 0.05 pour 5%)
        :param maturity: Maturité en années
        """
        super().__init__(face_value, maturity)
        self.yield_rate = yield_rate

    def price(self):
        """Calcul du prix d'un zéro coupon par actualisation de la valeur nominale."""
        return self.face_value / ((1 + self.yield_rate) ** self.maturity)


class FixedRateBond(Bond):
    def __init__(self, face_value, coupon_rate, maturity, frequency=1, yield_rate=None):
        """
        Obligation à taux fixe.

        :param face_value: Valeur nominale de l'obligation (ex: 1000)
        :param coupon_rate: Taux de coupon annuel (ex: 0.06 pour 6%)
        :param maturity: Maturité en années
        :param frequency: Nombre de paiements par an (ex: 2 pour semestriel)
        :param yield_rate: Taux de rendement (yield to maturity) annuel (ex: 0.05 pour 5%)
                           Doit être fourni pour actualiser les flux.
        """
        super().__init__(face_value, maturity)
        self.coupon_rate = coupon_rate
        self.frequency = frequency
        self.yield_rate = yield_rate

    def price(self):
        """Calcul du prix d'une obligation à taux fixe par actualisation des coupons et du principal."""
        if self.yield_rate is None:
            raise ValueError("Un taux de rendement (yield_rate) doit être fourni pour calculer le prix.")
        n_periods = int(self.maturity * self.frequency)
        coupon = self.face_value * self.coupon_rate / self.frequency
        price = sum([coupon / ((1 + self.yield_rate / self.frequency) ** (i + 1))
                     for i in range(n_periods)])
        price += self.face_value / ((1 + self.yield_rate / self.frequency) ** n_periods)
        return price


class FloatingRateBond(Bond):
    def __init__(self, face_value, margin, maturity, frequency=1, forecasted_rates=None, discount_rate=None):
        """
        Obligation à taux variable.

        :param face_value: Valeur nominale de l'obligation
        :param margin: Marge ajoutée au taux de référence (ex: 0.002 pour 0.2%)
        :param maturity: Maturité en années
        :param frequency: Nombre de paiements par an
        :param forecasted_rates: Liste des taux de référence prévisionnels pour chaque période (en décimal).
                                 Si None, on suppose un taux constant de 2%.
        :param discount_rate: Taux d'actualisation (flat) pour actualiser les flux.
                              Doit être fourni pour le calcul du prix.
        """
        super().__init__(face_value, maturity)
        self.margin = margin
        self.frequency = frequency
        self.discount_rate = discount_rate
        n_periods = int(maturity * frequency)
        if forecasted_rates is None:
            self.forecasted_rates = [0.02] * n_periods
        else:
            if len(forecasted_rates) != n_periods:
                raise ValueError("Le nombre de taux prévisionnels doit correspondre au nombre de périodes")
            self.forecasted_rates = forecasted_rates

    def price(self):
        """
        Calcul du prix d'une obligation à taux variable.
        Chaque coupon est égal à (taux de référence + marge) * (face_value / frequency)
        et le principal est remboursé à la maturité.
        """
        if self.discount_rate is None:
            raise ValueError("Un taux d'actualisation (discount_rate) doit être fourni pour calculer le prix.")
        n_periods = int(self.maturity * self.frequency)
        price = 0
        for i in range(n_periods):
            coupon_rate = self.forecasted_rates[i] + self.margin
            coupon = self.face_value * coupon_rate / self.frequency
            discount_factor = 1 / ((1 + self.discount_rate / self.frequency) ** (i + 1))
            price += coupon * discount_factor
        price += self.face_value / ((1 + self.discount_rate / self.frequency) ** n_periods)
        return price


# Exemple d'utilisation :

if __name__ == "__main__":
    # Obligation zéro coupon
    zcb = ZeroCouponBond(face_value=1000, yield_rate=0.05, maturity=5)
    print("Prix du Zero Coupon Bond :", round(zcb.price(), 2))

    # Obligation à taux fixe (paiement annuel)
    frb = FixedRateBond(face_value=1000, coupon_rate=0.06, maturity=5, frequency=1, yield_rate=0.05)
    print("Prix du Fixed Rate Bond :", round(frb.price(), 2))

    # Obligation à taux variable (paiement annuel)
    # Supposons des taux prévisionnels de 2%, 2.1%, 2.05%, 2.2% et 2.15% pour chaque année
    forecasted_rates = [0.02, 0.021, 0.0205, 0.022, 0.0215]
    varb = FloatingRateBond(face_value=1000, margin=0.002, maturity=5, frequency=1,
                            forecasted_rates=forecasted_rates, discount_rate=0.05)
    print("Prix du Floating Rate Bond :", round(varb.price(), 2))
