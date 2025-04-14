# products/concrete_strategies.py
from portfolio.strategy import Strategy
from option.option import Call, Put

# ---------------- Strategy Class Vanille ----------------
class BearCallSpread(Strategy):
    def __init__(self, strike_sell, strike_buy, maturity, exercise="european"):
        """
        Stratégie Bear Call Spread :
          - Vente d'un Call à strike bas (strike_sell)
          - Achat d'un Call à strike haut (strike_buy)
        :param strike_sell: Strike du Call vendu
        :param strike_buy: Strike du Call acheté
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Bear Call Spread")
        self.call_sell = Call(strike_sell, maturity, exercise)
        self.call_buy = Call(strike_buy, maturity, exercise)

    def get_legs(self):
        return [(self.call_sell, -1), (self.call_buy, 1)]


class PutCallSpread(Strategy):
    def __init__(self, strike, maturity, exercise="european"):
        """
        Stratégie Put-Call Spread :
          - Achat d'un Put
          - Vente d'un Call sur le même strike
        :param strike: Strike commun aux deux options
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Put-Call Spread")
        self.put = Put(strike, maturity, exercise)
        self.call = Call(strike, maturity, exercise)

    def get_legs(self):
        return [(self.put, 1), (self.call, -1)]


class BullCallSpread(Strategy):
    def __init__(self, strike_buy, strike_sell, maturity, exercise="european"):
        """
        Stratégie Bull Call Spread :
          - Achat d'un Call à strike bas (strike_buy)
          - Vente d'un Call à strike haut (strike_sell)
        :param strike_buy: Strike du Call acheté
        :param strike_sell: Strike du Call vendu
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Bull Call Spread")
        self.call_buy = Call(strike_buy, maturity, exercise)
        self.call_sell = Call(strike_sell, maturity, exercise)

    def get_legs(self):
        return [(self.call_buy, 1), (self.call_sell, -1)]


class ButterflySpread(Strategy):
    def __init__(self, strike_low, strike_mid, strike_high, maturity, exercise="european"):
        """
        Stratégie Butterfly Spread (Call) :
          - Achat d'un Call à strike bas (strike_low)
          - Vente de deux Calls à strike médian (strike_mid)
          - Achat d'un Call à strike haut (strike_high)
        :param strike_low: Strike du Call acheté à la baisse
        :param strike_mid: Strike du Call vendu (double position)
        :param strike_high: Strike du Call acheté à la hausse
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Butterfly Spread")
        self.call_low = Call(strike_low, maturity, exercise)
        self.call_mid = Call(strike_mid, maturity, exercise)
        self.call_high = Call(strike_high, maturity, exercise)

    def get_legs(self):
        return [(self.call_low, 1), (self.call_mid, -2), (self.call_high, 1)]


class Straddle(Strategy):
    def __init__(self, strike, maturity, exercise="european"):
        """
        Stratégie Straddle :
          - Achat simultané d'un Call et d'un Put sur le même strike
        :param strike: Strike commun aux deux options
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Straddle")
        self.call = Call(strike, maturity, exercise)
        self.put = Put(strike, maturity, exercise)

    def get_legs(self):
        return [(self.call, 1), (self.put, 1)]

# ---------------- Strategy Class Exotiques ----------------
class Strap(Strategy):
    def __init__(self, strike, maturity, exercise="european"):
        """
        Stratégie Strap :
          - Achat de deux Calls et d'un Put sur le même strike.
          Stratégie biaisée haussière.
        :param strike: Strike commun aux options
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Strap")
        self.call1 = Call(strike, maturity, exercise)
        self.call2 = Call(strike, maturity, exercise)
        self.put = Put(strike, maturity, exercise)

    def get_legs(self):
        return [(self.call1, 1), (self.call2, 1), (self.put, 1)]


class Strip(Strategy):
    def __init__(self, strike, maturity, exercise="european"):
        """
        Stratégie Strip :
          - Achat d'un Call et de deux Puts sur le même strike.
          Stratégie biaisée baissière.
        :param strike: Strike commun aux options
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Strip")
        self.call = Call(strike, maturity, exercise)
        self.put1 = Put(strike, maturity, exercise)
        self.put2 = Put(strike, maturity, exercise)

    def get_legs(self):
        return [(self.call, 1), (self.put1, 1), (self.put2, 1)]


class Strangle(Strategy):
    def __init__(self, lower_strike, upper_strike, maturity, exercise="european"):
        """
        Stratégie Strangle :
          - Achat d'un Put à un strike inférieur (lower_strike)
          - Achat d'un Call à un strike supérieur (upper_strike)
          Permet de profiter d'une forte volatilité sans biais directionnel.
        :param lower_strike: Strike du Put (inférieur)
        :param upper_strike: Strike du Call (supérieur)
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Strangle")
        self.put = Put(lower_strike, maturity, exercise)
        self.call = Call(upper_strike, maturity, exercise)

    def get_legs(self):
        return [(self.put, 1), (self.call, 1)]


class Condor(Strategy):
    def __init__(self, strike1, strike2, strike3, strike4, maturity, exercise="european"):
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
        :param maturity: Temps à l'échéance
        :param exercise: Type d'exercice ("european" par défaut)
        """
        super().__init__("Condor")
        self.call1 = Call(strike1, maturity, exercise)
        self.call2 = Call(strike2, maturity, exercise)
        self.call3 = Call(strike3, maturity, exercise)
        self.call4 = Call(strike4, maturity, exercise)

    def get_legs(self):
        return [(self.call1, 1), (self.call2, -1), (self.call3, -1), (self.call4, 1)]
