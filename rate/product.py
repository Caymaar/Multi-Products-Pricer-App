# ------------------- Bond Pricing -------------------


class ZeroCouponBond:
    def __init__(self, face_value, yield_rate, maturity):
        """
        face_value : valeur nominale (par exemple 1000)
        yield_rate : taux d'intérêt annuel (exprimé en décimal, par exemple 0.05 pour 5%)
        maturity   : maturité en années
        """
        self.face_value = face_value
        self.yield_rate = yield_rate
        self.maturity = maturity

    def price(self):
        """Prix d'un zéro coupon : actualisation de la valeur nominale"""
        return self.face_value / ((1 + self.yield_rate) ** self.maturity)


class FixedRateBond:
    def __init__(self, face_value, coupon_rate, maturity, frequency=1, yield_rate=None):
        """
        face_value   : valeur nominale de l'obligation (ex: 1000)
        coupon_rate  : taux de coupon annuel (ex: 0.06 pour 6%)
        maturity     : maturité en années
        frequency    : nombre de paiements par an (ex: 2 pour semestriel)
        yield_rate   : taux de rendement (yield to maturity) annuel (ex: 0.05 pour 5%)
                      Si yield_rate est fourni, le prix sera calculé par actualisation des flux.
        """
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.maturity = maturity
        self.frequency = frequency
        self.yield_rate = yield_rate

    def price(self):
        """Calcul du prix en actualisant les coupons et le remboursement final"""
        if self.yield_rate is None:
            raise ValueError("Un taux de rendement (yield_rate) doit être fourni pour calculer le prix.")
        n_periods = int(self.maturity * self.frequency)
        coupon = self.face_value * self.coupon_rate / self.frequency
        price = sum([coupon / ((1 + self.yield_rate / self.frequency) ** (i + 1))
                     for i in range(n_periods)])
        price += self.face_value / ((1 + self.yield_rate / self.frequency) ** n_periods)
        return price


class FloatingRateBond:
    def __init__(self, face_value, margin, maturity, frequency=1, forecasted_rates=None):
        """
        face_value      : valeur nominale de l'obligation
        margin          : marge ajoutée au taux de référence (ex: 0.002 pour 0.2%)
        maturity        : maturité en années
        frequency       : nombre de paiements par an
        forecasted_rates: liste des taux de référence prévisionnels (exprimés en décimal)
                          pour chaque période. Si None, on suppose un taux constant de 2% par exemple.
        """
        self.face_value = face_value
        self.margin = margin
        self.maturity = maturity
        self.frequency = frequency
        n_periods = int(maturity * frequency)
        if forecasted_rates is None:
            self.forecasted_rates = [0.02] * n_periods
        else:
            if len(forecasted_rates) != n_periods:
                raise ValueError("Le nombre de taux prévisionnels doit correspondre au nombre de périodes")
            self.forecasted_rates = forecasted_rates

    def price(self, discount_rate):
        """
        Calcul du prix de l'obligation à taux variable.
        discount_rate : taux de rendement (flat) pour actualiser les flux.
        Ici, chaque coupon est égal à (taux de référence + marge) * (face_value / frequency)
        et le remboursement du principal intervient à la maturité.
        """
        n_periods = int(self.maturity * self.frequency)
        price = 0
        for i in range(n_periods):
            coupon_rate = self.forecasted_rates[i] + self.margin
            coupon = self.face_value * coupon_rate / self.frequency
            discount_factor = 1 / ((1 + discount_rate / self.frequency) ** (i + 1))
            price += coupon * discount_factor
        price += self.face_value / ((1 + discount_rate / self.frequency) ** n_periods)
        return price


# Exemple d'utilisation :

if __name__ == "__main__":
    # Zéro coupon
    zcb = ZeroCouponBond(face_value=1000, yield_rate=0.05, maturity=5)
    print("Prix du Zero Coupon Bond :", round(zcb.price(), 2))

    # Obligation à taux fixe (paiement annuel)
    frb = FixedRateBond(face_value=1000, coupon_rate=0.06, maturity=5, frequency=1, yield_rate=0.05)
    print("Prix du Fixed Rate Bond :", round(frb.price(), 2))

    # Obligation à taux variable (paiement annuel)
    # Supposons des taux de référence prévisionnels de 2%, 2.1%, 2.05%, 2.2% et 2.15% pour chaque année
    forecasted_rates = [0.02, 0.021, 0.0205, 0.022, 0.0215]
    varb = FloatingRateBond(face_value=1000, margin=0.002, maturity=5, frequency=1, forecasted_rates=forecasted_rates)
    # Ici, on actualise avec un taux de 5%
    print("Prix du Floating Rate Bond :", round(varb.price(discount_rate=0.05), 2))
