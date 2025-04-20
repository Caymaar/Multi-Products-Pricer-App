from investment_strategies.strategy import Strategy
from datetime import datetime, timedelta
import pandas as pd
from market.market import Market
from pricers.mc_pricer import MonteCarloEngine
from rate.curve_utils import make_zc_curve
from rate.products import FixedRateBond, ZeroCouponBond
from option.option import Call, Put, DigitalCall, DigitalPut, DownAndOutPut


class StructuredStrategy(Strategy):
    def __init__(self, name):
        super().__init__(name)

    def get_legs(self):
        raise NotImplementedError("get_legs doit être implémentée dans chaque produit structuré.")


class ReverseConvertible(StructuredStrategy):
    def __init__(self, K, maturity, r):
        """
        Reverse convertible = ZCB + short put
        """
        super().__init__("Reverse Convertible")
        self.zcb = ZeroCouponBond(face_value=1000, maturity=maturity)
        self.put = Put(K, maturity)

    def get_legs(self):
        return [(self.zcb, 1), (self.put, -1)]


class TwinWin(StructuredStrategy):
    def __init__(self, K, maturity):
        """
        Twin win = long call + long put (symétrie autour du strike)
        """
        super().__init__("Twin Win")
        self.call = Call(K, maturity)
        self.put = Put(K, maturity)

    def get_legs(self):
        return [(self.call, 1), (self.put, 1)]


class BonusCertificate(StructuredStrategy):
    def __init__(self, K, barrier, maturity, r):
        """
        Bonus certificate = ZCB + Call - Put barrière knock-out
        """
        super().__init__("Bonus Certificate")
        self.zcb = ZeroCouponBond(face_value=1000, maturity=maturity)
        self.call = Call(K, maturity)
        self.barrier_put = DownAndOutPut(K=K, maturity=maturity, barrier=barrier)

    def get_legs(self):
        return [(self.zcb, 1), (self.call, 1), (self.barrier_put, -1)]


class LeverageCertificate(StructuredStrategy):
    def __init__(self, K, maturity, leverage=2):
        """
        Certificate offrant une exposition à effet de levier au sous-jacent :
        - financement partiel via un ZCB
        - achat d'un call multiplié par le facteur de levier
        """
        super().__init__("Leverage Certificate")
        # 1000 de nominal
        self.zcb = ZeroCouponBond(face_value=1000/leverage, maturity=maturity)
        self.call = Call(K, maturity)
        self.leverage = leverage

    def get_legs(self):
        # Le poids du call est égal au facteur de levier
        return [(self.zcb, 1), (self.call, self.leverage)]


class CappedParticipationCertificate(StructuredStrategy):
    def __init__(self, K, cap, maturity):
        """
        Certificate offrant participation jusqu'à un plafond cap :
        - ZCB pour la protection de capital
        - DigitalCall versant la performance jusqu'au cap
        """
        super().__init__("Capped Participation Certificate")
        self.zcb = ZeroCouponBond(face_value=1000, maturity=maturity)
        # paye (min(S_T - K, cap-K)) => digital call + vanilla call strip
        self.call = Call(K, maturity)
        self.digital = DigitalCall(K, maturity, "european", payoff=cap - K)

    def get_legs(self):
        # 1 ZCB + 1 call capte la hausse + 1 digital fixe à la limite
        return [(self.zcb, 1), (self.call, 1), (self.digital, 1)]


class DiscountCertificate(StructuredStrategy):
    def __init__(self, K, maturity):
        """
        Discount Certificate: verse le minimum(S_T, K)
        Payoff = K - Put(K, T)
        => équivalent à achat d'un ZCB + vente d'un put
        """
        super().__init__("Discount Certificate")
        self.zcb = ZeroCouponBond(face_value=K, maturity=maturity)
        self.put = Put(K, maturity)

    def get_legs(self):
        # La valeur finale = K - max(K - S_T, 0) = min(S_T, K)
        return [(self.zcb, 1), (self.put, -1)]


class ReverseConvertibleBarrier(StructuredStrategy):
    def __init__(self, K, barrier, maturity):
        """
        Reverse Convertible avec barrière knock-out:
        - ZCB
        - short Put
        - Knock-out Put (DownAndOutPut) pour limiter la perte
        """
        super().__init__("Reverse Convertible Barrier")
        self.zcb = ZeroCouponBond(face_value=1000, maturity=maturity)
        self.put = Put(K, maturity)
        self.down_and_out = DownAndOutPut(K=K, maturity=maturity, barrier=barrier)

    def get_legs(self):
        return [(self.zcb, 1), (self.put, -1), (self.down_and_out, 1)]


if __name__ == "__main__":
    # --- Dates ---
    pricing_date = datetime.today()
    maturity_years = 1  # horizon pour les produits structurés

    # --- Marché action pour les options ---
    S0 = 100
    r_stock = 0.05
    sigma = 0.2
    div = 0.0
    market_stock = Market(S0, r_stock, sigma, div, div_type="continuous")

    # --- Marché taux + courbe ZC ---
    data = pd.read_excel("../data_taux/RateCurve_temp.xlsx")
    maturities = data['Matu'].values
    rates = data['Rate'].values / 100
    zc_curve = make_zc_curve("interpolation", maturities, rates, kind="cubic")
    # modèle Vasicek si besoin pour futures extensions

    # --- Paramètres MC ---
    n_paths = 100000
    n_steps = 300
    seed = 42

    # --- Liste de produits structurés ---
    structured_products = [
        ReverseConvertible(K=100, maturity=maturity_years, r=r_stock),
        TwinWin(K=100, maturity=maturity_years),
        BonusCertificate(K=100, barrier=80, maturity=maturity_years, r=r_stock),
        LeverageCertificate(K=100, maturity=maturity_years, leverage=3),
        CappedParticipationCertificate(K=100, cap=120, maturity=maturity_years),
        DiscountCertificate(K=100, maturity=maturity_years),
        ReverseConvertibleBarrier(K=100, barrier=80, maturity=maturity_years)
    ]

    print("\n====== PRICING DES PRODUITS STRUCTURÉS ======\n")
    for prod in structured_products:
        total_price = 0.0
        details = []
        for leg, qty in prod.get_legs():
            # obligation
            if isinstance(leg, (FixedRateBond, ZeroCouponBond)):
                price_leg = leg.price(zc_curve)
            else:
                # option via MC European
                engine = MonteCarloEngine(
                    market_stock, leg, pricing_date,
                    n_paths, n_steps, seed=seed
                )
                price_leg = engine.price(type="MC")
            details.append((leg.__class__.__name__, qty, price_leg))
            total_price += qty * price_leg

        print(f"=== {prod.name} ===")
        for name, qty, price in details:
            print(f"{name:>25} | qty: {qty:+} | price: {price:.4f}")
        print(f"→ Prix total {prod.name}: {total_price:.4f}\n")


