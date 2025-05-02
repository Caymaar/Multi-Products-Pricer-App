from investment_strategies.abstract_strategy import Strategy
from option.option import OptionPortfolio, Call, Put
from pricers.mc_pricer import MonteCarloEngine
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

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
        assert strike_sell < strike_buy, \
            "Pour un Bear Call Spread, strike_sell < strike_buy"
        super().__init__("Bear Call Spread", pricing_date, maturity_date, convention_days)
        self.call_sell = Call(strike_sell, maturity_date, exercise)
        self.call_buy = Call(strike_buy, maturity_date, exercise)
        self._populate_legs()

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
        # seuls call et put au même strike
        assert isinstance(strike, (int, float)), "Strike doit être numérique"
        super().__init__("Put-Call Spread", pricing_date, maturity_date, convention_days)
        self.put = Put(strike, maturity_date, exercise)
        self.call = Call(strike, maturity_date, exercise)
        self._populate_legs()

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
        assert strike_buy < strike_sell, \
            "Pour un Bull Call Spread, strike_buy < strike_sell"
        super().__init__("Bull Call Spread", pricing_date, maturity_date, convention_days)
        self.call_buy = Call(strike_buy, maturity_date, exercise)
        self.call_sell = Call(strike_sell, maturity_date, exercise)
        self._populate_legs()

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
        assert strike_mid - strike_low == strike_high - strike_mid, \
            "Pour un butterfly symétrique, K_mid - K_low doit = K_high - K_mid"
        super().__init__("Butterfly Spread", pricing_date, maturity_date, convention_days)
        self.call_low = Call(strike_low, maturity_date, exercise)
        self.call_mid = Call(strike_mid, maturity_date, exercise)
        self.call_high = Call(strike_high, maturity_date, exercise)
        self._populate_legs()

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
        self._populate_legs()

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
        self.call = Call(strike, maturity_date, exercise)
        self.put = Put(strike, maturity_date, exercise)
        self._populate_legs()

    def get_legs(self):
        return [(self.call, 2), (self.put, 1)]


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
        self.put = Put(strike, maturity_date, exercise)
        self._populate_legs()

    def get_legs(self):
        return [(self.call, 1), (self.put, 2)]


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
        self._populate_legs()

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
        assert strike2 - strike1 == strike4 - strike3, \
            "Pour un condor symétrique, K2 - K1 doit = K4 - K3"
        super().__init__("Condor", pricing_date, maturity_date, convention_days)
        self.call1 = Call(strike1, maturity_date, exercise)
        self.call2 = Call(strike2, maturity_date, exercise)
        self.call3 = Call(strike3, maturity_date, exercise)
        self.call4 = Call(strike4, maturity_date, exercise)
        self._populate_legs()

    def get_legs(self):
        return [(self.call1, 1), (self.call2, -1), (self.call3, -1), (self.call4, 1)]


def plot_strategy_payoff(strategy, S_range=None, buffer: float = 0.5):
    """
    Trace le profil de payoff à maturité pour une stratégie.
    Si S_range n'est pas fourni, on le déduit des strikes :
      - on récupère tous les strikes des options de la stratégie,
      - on prend [min_strike*(1-buffer), max_strike*(1+buffer)].
    :param strategy: instance de Strategy (avec get_legs())
    :param S_range: tableau de prix sous-jacent à maturité à tester
    :param buffer: proportion autour des strikes pour étendre la grille
    """

    global strikes

    # 1) Détermination de la plage de S si non fournie
    if S_range is None:
        strikes = []
        for opt, _ in strategy.get_legs():
            if hasattr(opt, "K"):
                strikes.append(opt.K)
        if strikes:
            k_min = min(strikes)
            k_max = max(strikes)
            low  = max(0.0, k_min * (1 - buffer))
            high = k_max * (1 + buffer)
        else:
            # fallback si pas d'attribut K (produits non-standard)
            low, high = 0.0, strategy.notional * (1 + buffer)
        S_range = np.linspace(low, high, 500)

    # 2) Calcul du payoff net
    payoffs = np.zeros_like(S_range)
    for option, qty in strategy.get_legs():
        intrinsic = np.array([option.intrinsic_value(ST) for ST in S_range])
        payoffs += qty * intrinsic

    # 3) Tracé
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(S_range, payoffs, lw=2, label=strategy.name)
    ax.fill_between(S_range, payoffs, 0, where=(payoffs >= 0),
                    alpha=0.5, label="Gain")
    ax.fill_between(S_range, payoffs, 0, where=(payoffs <  0),
                    alpha=0.4, label="Perte")

    # Repère au niveau du ou des strikes
    for K in set(strikes):
        ax.axvline(K, color="gray", linestyle="--", lw=1, alpha=0.7)

    ax.axhline(0, color="black", lw=1)
    ax.set_title(f"Payoff à maturité – {strategy.name}", fontsize=14, weight="bold")
    ax.set_xlabel("Prix du sous-jacent à maturité", fontsize=12)
    ax.set_ylabel("Payoff net", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="best", frameon=False)
    plt.tight_layout()

    return fig


if __name__ == "__main__":

    from market.market_factory import create_market

    # === 1) Définir la date de pricing et la maturité (5 ans) ===
    pricing_date = datetime(2023, 4, 25)
    maturity_date = datetime(2028, 4, 25)

    # === 2) Paramètres pour Svensson ===
    sv_guess = [0.02, -0.01, 0.01, 0.005, 1.5, 3.5]
    # === 3) Instanciation « tout‐en‐un » du Market LVMH ===
    market = create_market(
        stock="LVMH",
        pricing_date=pricing_date,
        vol_source="implied",  # ou "historical"
        hist_window=252,
        curve_method="svensson",  # méthode de calibration
        curve_kwargs={"initial_guess": sv_guess},
        dcc="Actual/Actual",
    )

    K = market.S0 * 0.9

    # barrière “up” à 120% de S0,
    # barrière “down” à 80% de S0
    barrier_up = market.S0 * 1.2
    barrier_down = market.S0 * 0.8

    # --- Liste de stratégies à tester ---
    strategies = [
        BearCallSpread(strike_sell=K, strike_buy=K + 30, pricing_date=pricing_date, maturity_date=maturity_date),
        BullCallSpread(strike_buy=K, strike_sell=K + 30, pricing_date=pricing_date, maturity_date=maturity_date),
        ButterflySpread(strike_low=K, strike_mid=K + 30, strike_high=K + 60, pricing_date=pricing_date,
                        maturity_date=maturity_date),
        Straddle(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
        Strap(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
        Strip(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
        Strangle(lower_strike=K, upper_strike=K + 60, pricing_date=pricing_date, maturity_date=maturity_date),
        Condor(strike1=K, strike2=K + 15, strike3=K + 50, strike4=K + 65, pricing_date=pricing_date,
               maturity_date=maturity_date),
        PutCallSpread(strike=K, pricing_date=pricing_date, maturity_date=maturity_date),
    ]


    def price_vanilla_strategy(
            strategy,
            market,
            pricing_date,
            n_paths: int,
            n_steps: int,
            seed: int = None,
            alpha: float = 0.05,
            pricing_type: str = "MC",
            exercise: str = "european"
    ) -> tuple[float, float, float]:
        """
        Prix Monte-Carlo d'une stratégie vanille en une passe.
        Retourne (prix, CI_bas, CI_haut).
        """
        # 1) Récupère toutes les jambes et leurs quantités
        legs, qtys = zip(*strategy.get_legs())
        # 2) Pack dans un seul OptionPortfolio en spécifiant le type d'exercice
        for leg in legs:
            leg.exercise = exercise

        ptf = OptionPortfolio(list(legs), list(qtys))
        # 3) Instancie l'engine sur TOUT le portefeuille
        engine = MonteCarloEngine(
            market=market,
            option_ptf=ptf,
            pricing_date=pricing_date,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed
        )
        # 4) Calcule prix et intervalle
        prices = engine.price(type=pricing_type)  # vecteur de prix par option
        ci_low, ci_up = engine.price_confidence_interval(type="MC", alpha=alpha)
        # 5) Somme pondérée
        total_price = float(np.dot(prices, qtys))
        total_low = float(np.dot(ci_low, qtys))
        total_up = float(np.dot(ci_up, qtys))

        plot_strategy_payoff(strategy)

        return total_price, total_low, total_up

    # --- Paramètres Modèle ---
    n_paths = 100000
    n_steps = 300
    seed = 2

    print("\n====== EUROPEAN VANILLA STRATEGIES PRICING ======")

    for strat in strategies:
        p, low, up = price_vanilla_strategy(
            strategy=strat,
            market=market,
            pricing_date=pricing_date,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            alpha=0.05,
            pricing_type="MC",
            exercise="european"
        )
        print(f"{strat.name:20s} → Prix: {p:.4f} | CI95%: [{low:.4f}, {up:.4f}]")

    print("\n====== EUROPEAN AMERICAN STRATEGIES PRICING ======")

    for strat in strategies:
        p, low, up = price_vanilla_strategy(
            strategy=strat,
            market=market,
            pricing_date=pricing_date,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=seed,
            alpha=0.05,
            pricing_type="Longstaff",
            exercise="american"
        )
        print(f"{strat.name:20s} → Prix: {p:.4f} | CI95%: [{low:.4f}, {up:.4f}]")