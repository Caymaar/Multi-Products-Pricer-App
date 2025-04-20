from investment_strategies.strategy import Strategy
from option.option import Call, Put
from market.market import Market
from pricers.mc_pricer import MonteCarloEngine
from datetime import datetime, timedelta

# ---------------- Strategy Class Vanille ----------------
class BearCallSpread(Strategy):
    def __init__(self, strike_sell, strike_buy, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Bear Call Spread :
          - Vente d'un Call à strike bas (strike_sell)
          - Achat d'un Call à strike haut (strike_buy)
        :param strike_sell: Strike du Call vendu
        :param strike_buy: Strike du Call acheté
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Bear Call Spread", pricing_date, maturity_date, convention_days)
        self.call_sell = Call(strike_sell, maturity_date, exercise)
        self.call_buy = Call(strike_buy, maturity_date, exercise)

    def get_legs(self):
        return [(self.call_sell, -1), (self.call_buy, 1)]


class PutCallSpread(Strategy):
    def __init__(self, strike, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Put-Call Spread :
          - Achat d'un Put
          - Vente d'un Call sur le même strike
        :param strike: Strike commun aux deux options
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Put-Call Spread", pricing_date, maturity_date, convention_days)
        self.put = Put(strike, maturity_date, exercise)
        self.call = Call(strike, maturity_date, exercise)

    def get_legs(self):
        return [(self.put, 1), (self.call, -1)]


class BullCallSpread(Strategy):
    def __init__(self, strike_buy, strike_sell, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Bull Call Spread :
          - Achat d'un Call à strike bas (strike_buy)
          - Vente d'un Call à strike haut (strike_sell)
        :param strike_buy: Strike du Call acheté
        :param strike_sell: Strike du Call vendu
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Bull Call Spread", pricing_date, maturity_date, convention_days)
        self.call_buy = Call(strike_buy, maturity_date, exercise)
        self.call_sell = Call(strike_sell, maturity_date, exercise)

    def get_legs(self):
        return [(self.call_buy, 1), (self.call_sell, -1)]


class ButterflySpread(Strategy):
    def __init__(self, strike_low, strike_mid, strike_high, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Butterfly Spread (Call) :
          - Achat d'un Call à strike bas (strike_low)
          - Vente de deux Calls à strike médian (strike_mid)
          - Achat d'un Call à strike haut (strike_high)
        :param strike_low: Strike du Call acheté à la baisse
        :param strike_mid: Strike du Call vendu (double position)
        :param strike_high: Strike du Call acheté à la hausse
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Butterfly Spread", pricing_date, maturity_date, convention_days)
        self.call_low = Call(strike_low, maturity_date, exercise)
        self.call_mid = Call(strike_mid, maturity_date, exercise)
        self.call_high = Call(strike_high, maturity_date, exercise)

    def get_legs(self):
        return [(self.call_low, 1), (self.call_mid, -2), (self.call_high, 1)]


class Straddle(Strategy):
    def __init__(self, strike, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Straddle :
          - Achat simultané d'un Call et d'un Put sur le même strike
        :param strike: Strike commun aux deux options
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Straddle", pricing_date, maturity_date, convention_days)
        self.call = Call(strike, maturity_date, exercise)
        self.put = Put(strike, maturity_date, exercise)

    def get_legs(self):
        return [(self.call, 1), (self.put, 1)]

# ---------------- Strategy Class Exotiques ----------------
class Strap(Strategy):
    def __init__(self, strike, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Strap :
          - Achat de deux Calls et d'un Put sur le même strike.
          Stratégie biaisée haussière.
        :param strike: Strike commun aux options
        :param maturity_date: Temps à l'échéance
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Strap", pricing_date, maturity_date, convention_days)
        self.call1 = Call(strike, maturity_date, exercise)
        self.call2 = Call(strike, maturity_date, exercise)
        self.put = Put(strike, maturity_date, exercise)

    def get_legs(self):
        return [(self.call1, 1), (self.call2, 1), (self.put, 1)]


class Strip(Strategy):
    def __init__(self, strike, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Strip :
          - Achat d'un Call et de deux Puts sur le même strike.
          Stratégie biaisée baissière.
        :param strike: Strike commun aux options
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Strip", pricing_date, maturity_date, convention_days)
        self.call = Call(strike, maturity_date, exercise)
        self.put1 = Put(strike, maturity_date, exercise)
        self.put2 = Put(strike, maturity_date, exercise)

    def get_legs(self):
        return [(self.call, 1), (self.put1, 1), (self.put2, 1)]


class Strangle(Strategy):
    def __init__(self, lower_strike, upper_strike, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Strangle :
          - Achat d'un Put à un strike inférieur (lower_strike)
          - Achat d'un Call à un strike supérieur (upper_strike)
          Permet de profiter d'une forte volatilité sans biais directionnel.
        :param lower_strike: Strike du Put (inférieur)
        :param upper_strike: Strike du Call (supérieur)
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Strangle", pricing_date, maturity_date, convention_days)
        self.put = Put(lower_strike, maturity_date, exercise)
        self.call = Call(upper_strike, maturity_date, exercise)

    def get_legs(self):
        return [(self.put, 1), (self.call, 1)]


class Condor(Strategy):
    def __init__(self, strike1, strike2, strike3, strike4, pricing_date, maturity_date, exercise="european", convention_days="Actual/365"):
        """
        Stratégie Condor (Call) :
          - Achat d'un Call à strike1
          - Vente d'un Call à strike2
          - Vente d'un Call à strike3
          - Achat d'un Call à strike4
        :param strike1: Strike du Call acheté (le plus bas)
        :param strike2: Strike du premier Call vendu
        :param strike3: Strike du second Call vendu
        :param strike4: Strike du Call acheté (le plus haut)
        :param maturity_date: Date de maturité
        :param pricing_date: Date de pricing
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Condor", pricing_date, maturity_date, convention_days)
        self.call1 = Call(strike1, maturity_date, exercise)
        self.call2 = Call(strike2, maturity_date, exercise)
        self.call3 = Call(strike3, maturity_date, exercise)
        self.call4 = Call(strike4, maturity_date, exercise)

    def get_legs(self):
        return [(self.call1, 1), (self.call2, -1), (self.call3, -1), (self.call4, 1)]

if __name__ == "__main__":

    # --- Date de pricing et maturité à +1 an ---
    pricing_date = datetime.today()
    maturity_date = pricing_date + timedelta(days=365)

    # --- Paramètres marché ---
    S0 = 100
    r = 0.05
    sigma = 0.2
    div = 0.0
    K = 100

    market = Market(S0, r, sigma, div, div_type="continuous")

    # --- Liste de stratégies à tester ---
    strategies = [
        BearCallSpread(strike_sell=95, strike_buy=105, pricing_date=pricing_date, maturity_date=maturity_date),
        BullCallSpread(strike_buy=95, strike_sell=105, pricing_date=pricing_date, maturity_date=maturity_date),
        ButterflySpread(strike_low=90, strike_mid=100, strike_high=110, pricing_date=pricing_date, maturity_date=maturity_date),
        Straddle(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
        Strap(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
        Strip(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
        Strangle(lower_strike=90, upper_strike=110, pricing_date=pricing_date, maturity_date=maturity_date),
        Condor(strike1=90, strike2=95, strike3=105, strike4=110, pricing_date=pricing_date, maturity_date=maturity_date),
        PutCallSpread(strike=100, pricing_date=pricing_date, maturity_date=maturity_date),
    ]

    # --- Paramètres Modèle ---
    n_paths = 100000
    n_steps = 300
    seed = 2

    print("\n====== EUROPEAN VANILLA STRATEGIES PRICING ======")

    # --- Pricing des stratégies à caractère européen ---
    for strat in strategies:
        print(f"\n=== {strat.name} ===")

        price = 0
        prices = []

        for leg, quantity in strat.get_legs():
            engine = MonteCarloEngine(
                market=market,
                option=leg,
                pricing_date=pricing_date,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed
            )

            p = engine.price(type="MC")
            ci_low, ci_up = engine.price_confidence_interval(type="MC")

            prices.append((leg.__class__.__name__, p, ci_low, ci_up, quantity))
            price += quantity * p

        for leg_name, p, low, high, q in prices:
            print(f"{leg_name:>15} | Quantité: {q:+} | Prix: {p:.4f} | CI95%: [{low:.4f}, {high:.4f}]")

        print(f"→ Prix total stratégie : {price:.4f}")

    # --- Pricing des stratégies américaines ---
    print("\n====== AMERICAN VANILLA STRATEGIES PRICING ======")

    for strat in strategies:
        print(f"\n=== {strat.name} (American) ===")

        price = 0
        prices = []

        for leg, quantity in strat.get_legs():
            # Assure que chaque leg est bien en "american"
            leg.exercise = "american"

            engine = MonteCarloEngine(
                market=market,
                option=leg,
                pricing_date=pricing_date,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed
            )

            try:
                p = engine.price(type="Longstaff")
                ci_low, ci_up = engine.price_confidence_interval(type="Longstaff")

                prices.append((leg.__class__.__name__, p, ci_low, ci_up, quantity))
                price += quantity * p

            except NotImplementedError as e:
                print(f"NotImplementedError for {leg.__class__.__name__}: {e}")

        for leg_name, p, low, high, q in prices:
            print(f"{leg_name:>15} | Quantité: {q:+} | Prix: {p:.4f} | CI95%: [{low:.4f}, {high:.4f}]")

        print(f"→ Prix total stratégie (Américaine) : {price:.4f}")