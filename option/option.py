import numpy as np
from abc import ABC, abstractmethod
from typing import List

# Note d'avertissement :
# Les options vanilles et digitales, qu'elles soient de style américain ou européen, seront valorisées par n'importe quelle méthode de pricing disponible (Trinomial ou MC).
# Cependant, les options barrières seront valorisées par Monte Carlo pour les européennes et Longstaff pour les américaines, impossible avec Trinomial en raison de la recombinaison tout en ajustant les probabilités et la structure.

# ---------------- Base Option Class ----------------
class Option(ABC):
    def __init__(self, K, maturity, exercise="european"):
        self.K = K  # Strike price
        self.T = maturity  # Time to maturity
        self.exercise = exercise.lower()  # "european", "american", etc.
        self.name = f'{self.__class__.__name__}, K={self.K}, T={self.T}, exercise={self.exercise}'

    @abstractmethod
    def intrinsic_value(self, S):
        """
        Méthode abstraite pour calculer la valeur intrinsèque d'une option,
        S représentant les prix pour différents chemins.
        """
        raise NotImplementedError("La méthode intrinsic_value doit être implémentée dans les sous-classes.")

    def _terminal_price(self, S):
        # Si S est un scalaire (float ou int), on retourne S directement (utile notamment pour le pricing trinomial par noeud)
        if np.isscalar(S):
            return S
        # Sinon, on suppose que S est une matrice, on récupère la dernière colonne
        return S[:, -1]

# ---------------- Option Portfolio Class ----------------
class OptionPortfolio:
    def __init__(self, options: Option | List[Option], weights : List[float] = None):
        """
        Initialise un portefeuille d'options.
        :param options: Liste d'options à inclure dans le portefeuille.
        """
        self.assets = options
        self.weights = weights if weights is not None else [1]*len(options)
        if len(self.assets) != len(self.weights):
            raise ValueError("La taille des actifs du portefeuille doit être égale aux poids alloués !")


    def intrinsic_value(self, S_slices):
        """
        Calcule la valeur intrinsèque du portefeuille d'options.
        :param S_slicDownAndOutCalles: Matrice des prix simulés pour chaque option.
        :return: Valeur totale du portefeuille d'options.
        """
        total_value = 0
        for option, S in zip(self.assets, S_slices):
            total_value += option.intrinsic_value(S)
        return total_value



# ---------------- Call and Put Option Classes ----------------
class Call(Option):
    def __init__(self, K, maturity, exercise="european"):
        super().__init__(K, maturity, exercise)

    def intrinsic_value(self, S):
        """ Payoff d'un Call à la période observée. """
        S_T = self._terminal_price(S)
        return np.maximum(S_T - self.K, 0)


class Put(Option):
    def __init__(self, K, maturity, exercise="european"):
        super().__init__(K, maturity, exercise)

    def intrinsic_value(self, S):
        """ Payoff d'un Put à la période observée. """
        S_T = self._terminal_price(S)
        return np.maximum(self.K - S_T, 0)


# ---------------- Digital Option Classes ----------------
class DigitalCall(Call):
    def __init__(self, K, maturity, exercise, payoff):
        """
        Digital Call : option digitale cash-or-nothing call.
        Paye un montant fixe (payoff) si le prix terminal est supérieur à K.
        :param K: Strike
        :param maturity: Maturité
        :param exercise: Type d'exercice (par défaut "european")
        :param payoff: Montant payé si l'option est dans la monnaie (par défaut 1.0)
        """
        super().__init__(K, maturity, exercise)
        self.payoff_amount = payoff

    def intrinsic_value(self, S):
        """
        Si le prix terminal S > K, renvoie payoff_amount, sinon 0.
        Utilise np.where pour une compatibilité vectorielle.
        """
        S_T = self._terminal_price(S)
        return np.where(S_T > self.K, self.payoff_amount, 0)


class DigitalPut(Put):
    def __init__(self, K, maturity, exercise="european", payoff=1.0):
        """
        Digital Put : option digitale cash-or-nothing put.
        Paye un montant fixe (payoff) si le prix terminal est inférieur à K.
        :param K: Strike
        :param maturity: Maturité
        :param exercise: Type d'exercice (par défaut "european")
        :param payoff: Montant payé si l'option est dans la monnaie (par défaut 1.0)
        """
        super().__init__(K, maturity, exercise)
        self.payoff_amount = payoff

    def intrinsic_value(self, S):
        """
        Si le prix terminal S < K, renvoie payoff_amount, sinon 0.
        """
        S_T = self._terminal_price(S)
        return np.where(S_T < self.K, self.payoff_amount, 0)


# ---------------- Barrier Option Classes -----------------------------
class BarrierOption(Option):
    def __init__(self, K, maturity, exercise, barrier, direction, knock_type, rebate=0):
        """
        Classe de base pour les options barrières.
        Une option barrière a un prix de barrière et peut être "knock-in" ou "knock-out".

        :param K: Strike
        :param maturity: Maturité
        :param exercise: Type d'exercice (par défaut "european")
        :param barrier: Niveau de la barrière
        :param direction: "up" ou "down" pour la direction de la barrière
        :param knock_type: "in" ou "out" pour déterminer si l'option est activée ou désactivée
        :param rebate: Montant payé si l'option est activée mais ne touche pas la barrière
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier
        self.direction = direction  # "up" ou "down"
        self.knock_type = knock_type  # "in" ou "out"
        self.rebate = rebate
        self.name = f'{self.__class__.__name__}, K={self.K}, T={self.T}, exercise={self.exercise}'

    def is_barrier_triggered(self, S):
        """
        Vérifie si la barrière a été atteinte.
        Renvoie un tableau booléen indiquant si la barrière a été touchée sur chaque chemin.

        :param S: Les prix simulés
        :return: True si la barrière est touchée, False sinon
        """
        if self.direction == "up":
            return np.max(S, axis=1) >= self.barrier
        elif self.direction == "down":
            return np.min(S, axis=1) <= self.barrier
        else:
            raise ValueError("La direction doit être 'up' ou 'down'.")

    def intrinsic_value(self, S):
        """
        Calcule la valeur intrinsèque de l'option barrière.

        :param S: Les prix simulés
        :return: La valeur de l'option en fonction de si la barrière est touchée ou non.
        """
        triggered = self.is_barrier_triggered(S)
        base_payoff = super().intrinsic_value(S)

        if self.knock_type == "in":
            return np.where(triggered, base_payoff, self.rebate)
        elif self.knock_type == "out":
            return np.where(triggered, self.rebate, base_payoff)
        else:
            raise ValueError("knock_type doit être 'in' ou 'out'")


# -------- Knock-Out Options --------
class UpAndOutCall(BarrierOption, Call):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Up-and-Out Call : Call avec barrière supérieure.
        L'option est désactivée si le prix touche ou dépasse la barrière (up).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="out", direction="up", rebate=rebate)


class DownAndOutCall(BarrierOption, Call):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Down-and-Out Call : Call avec barrière inférieure.
        L'option est désactivée si le prix touche ou descend en dessous de la barrière (down).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="out", direction="down", rebate=rebate)


class UpAndOutPut(BarrierOption, Put):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Up-and-Out Put : Put avec barrière supérieure.
        L'option est désactivée si le prix touche ou dépasse la barrière (up).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="out", direction="up", rebate=rebate)


class DownAndOutPut(BarrierOption, Put):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Down-and-Out Put : Put avec barrière inférieure.
        L'option est désactivée si le prix touche ou descend en dessous de la barrière (down).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="out", direction="down", rebate=rebate)


# -------- Knock-In Options --------
class UpAndInCall(BarrierOption, Call):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Up-and-In Call : Call avec barrière supérieure.
        L'option est activée uniquement si le prix touche ou dépasse la barrière (up).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="in", direction="up", rebate=rebate)


class DownAndInCall(BarrierOption, Call):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Down-and-In Call : Call avec barrière inférieure.
        L'option est activée uniquement si le prix touche ou descend en dessous de la barrière (down).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="in", direction="down", rebate=rebate)


class UpAndInPut(BarrierOption, Put):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Up-and-In Put : Put avec barrière supérieure.
        L'option est activée uniquement si le prix touche ou dépasse la barrière (up).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="in", direction="up", rebate=rebate)


class DownAndInPut(BarrierOption, Put):
    def __init__(self, K, maturity, barrier, rebate=0, exercise="european"):
        """
        Option Down-and-In Put : Put avec barrière inférieure.
        L'option est activée uniquement si le prix touche ou descend en dessous de la barrière (down).
        """
        super().__init__(K, maturity, exercise, barrier, knock_type="in", direction="down", rebate=rebate)

if __name__ == '__main__':

    from pricers.mc_pricer import MonteCarloEngine
    from pricers.tree_pricer import TreePortfolio
    from datetime import datetime
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

    K = market_lvmh.S0 * 0.9

    # barrière “up” à 120% de S0,
    # barrière “down” à 80% de S0
    barrier_up = market_lvmh.S0 * 1.2
    barrier_down = market_lvmh.S0 * 0.8

    # Options testables avec l'arbre trinomial (pas de barrières ici)
    options_tree = OptionPortfolio([
        Call(K, maturity_date, exercise="european"),
        Put(K, maturity_date, exercise="european"),
        DigitalCall(K, maturity_date, exercise="european", payoff=10.0),
        DigitalPut(K, maturity_date, exercise="european", payoff=10.0),
        Call(K, maturity_date, exercise="american"),
        Put(K, maturity_date, exercise="american"),
        DigitalCall(K, maturity_date, exercise="american", payoff=10.0),
        DigitalPut(K, maturity_date, exercise="american", payoff=10.0),
    ])

    # --- Paramètres ---
    n_paths = 10000
    n_steps = 300
    seed = 2

    print("\n====== TRINOMIAL TREE PRICING ======")

    engine = TreePortfolio(
        market=market_lvmh,
        option_ptf=options_tree,
        pricing_date=pricing_date,
        n_steps=n_steps
    )

    price = engine.price()
    print(f"Prix estimé (Trinomial Tree) : {np.round(price, 4)}")

    options = OptionPortfolio([
        Call(K, maturity_date, exercise="european"),
        Put(K, maturity_date, exercise="european"),
        DigitalCall(K, maturity_date, exercise="european", payoff=10.0),
        DigitalPut(K, maturity_date, exercise="european", payoff=10.0),

        # barrier‐options : barrère au‐dessus de S0 pour les “up”, en dessous pour les “down”
        UpAndOutCall(K, maturity_date, barrier=barrier_up, rebate=0, exercise="european"),
        UpAndInCall(K, maturity_date, barrier=barrier_up, rebate=0, exercise="european"),
        DownAndOutPut(K, maturity_date, barrier=barrier_down, rebate=0, exercise="european"),
        DownAndInPut(K, maturity_date, barrier=barrier_down, rebate=0, exercise="european"),
    ])

    print("\n====== EUROPEAN MONTE CARLO PRICING ======")

    engine = MonteCarloEngine(
        market=market_lvmh,
        option_ptf=options,
        pricing_date=datetime.today(),
        n_paths=n_paths,
        n_steps=n_steps,
        seed=seed
    )

    price = engine.price(type="MC")
    ci_low, ci_up = engine.price_confidence_interval(type="MC")

    print(f"Prix estimé      : {np.round(price, 4)}")
    print(f"Intervalle 95%   : [{np.round(ci_low, 4)}, {np.round(ci_up, 4)}]")