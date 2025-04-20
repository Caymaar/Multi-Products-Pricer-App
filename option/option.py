import numpy as np
from abc import ABC, abstractmethod
from typing import List

# Warning note :
# Vanilla & digits options, both american or european style will be priced by any available pricing method (Trinomial or MC)
# However, Barrier option will be priced by Monte Carlo for european & Longstaff for American, not possible for Trinomial being recombined while adjusting probas and structure

# ---------------- Base Option Class ----------------
class Option(ABC):
    def __init__(self, K, maturity, exercise="european"):
        self.K = K  # Strike price
        self.T = maturity  # Time to maturity
        self.exercise = exercise.lower()  # "european", "american", etc.
        self.name = None

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
    def __init__(self, options : Option | List[Option]):
        """
        Initialise un portefeuille d'options.
        :param options: Liste d'options à inclure dans le portefeuille.
        """
        self.options = options

    def intrinsic_value(self, S_slices):
        """
        Calcule la valeur intrinsèque du portefeuille d'options.
        :param S_slices: Matrice des prix simulés pour chaque option.
        :return: Valeur totale du portefeuille d'options.
        """
        total_value = 0
        for option, S in zip(self.options, S_slices):
            total_value += option.intrinsic_value(S)
        return total_value


# ---------------- Call and Put Option Classes ----------------
class Call(Option):
    def __init__(self, K, maturity, exercise="european"):
        super().__init__(K, maturity, exercise)
        self.name = f'{self.__class__.__name__}, K={self.K}, T={self.T.date()}, exercise={self.exercise}'

    def intrinsic_value(self, S):
        """ Payoff d'un Call à la période observée. """
        S_T = self._terminal_price(S)
        return np.maximum(S_T - self.K, 0)


class Put(Option):
    def __init__(self, K, maturity, exercise="european"):
        super().__init__(K, maturity, exercise)
        self.name = f'{self.__class__.__name__}, K={self.K}, T={self.T.date()}, exercise={self.exercise}'

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
        self.name = f'{self.__class__.__name__}, K={self.K}, T={self.T.date()}, exercise={self.exercise}'

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
