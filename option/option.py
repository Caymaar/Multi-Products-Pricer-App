import numpy as np
from abc import ABC, abstractmethod

# ---------------- Base Option Class ----------------
class Option(ABC):
    def __init__(self, K, maturity, exercise="european"):
        self.K = K              # Strike price
        self.T = maturity       # Time to maturity
        self.exercise = exercise.lower()  # "european", "american", etc.

    @abstractmethod
    def payoff(self, S):
        """ Méthode abstraite pour calculer le payoff d'une option à maturité,
            S représentant soit le prix terminal, soit le chemin complet. """
        raise NotImplementedError("La méthode payoff doit être implémentée dans les sous-classes.")

# ---------------- Call and Put Option Classes ----------------
class Call(Option):
    def __init__(self, K, maturity, exercise="european"):
        super().__init__(K, maturity, exercise)

    def payoff(self, S):
        """ Payoff d'un Call à maturité. """
        return np.maximum(S[-1] - self.K, 0)

class Put(Option):
    def __init__(self, K, maturity, exercise="european"):
        super().__init__(K, maturity, exercise)

    def payoff(self, S):
        """ Payoff d'un Put à maturité. """
        return np.maximum(self.K - S[-1], 0)


# ---------------- Digital Option Classes ----------------
class DigitalCall(Option):
    def __init__(self, K, maturity, exercise="european", payoff=1.0):
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

    def payoff(self, S):
        """
        Si le prix terminal S[-1] > K, renvoie payoff_amount, sinon 0.
        Utilise np.where pour une compatibilité vectorielle.
        """
        return np.where(S[-1] > self.K, self.payoff_amount, 0)
    # ou on peut faire la moyenne des payoffs pour chaque chemin
    #        indicator = np.where(S > self.K, 1.0, 0.0)
    #     avg_indicator = np.mean(indicator)
    #     return self.payoff_amount * avg_indicator


class DigitalPut(Option):
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

    def payoff(self, S):
        """
        Si le prix terminal S[-1] < K, renvoie payoff_amount, sinon 0.
        """
        return np.where(S[-1] < self.K, self.payoff_amount, 0)
    # ou on peut faire la moyenne des payoffs pour chaque chemin
    # indicator = np.where(S < self.K, 1.0, 0.0)
    # avg_indicator = np.mean(indicator)
    # return self.payoff_amount * avg_indicator


# ---------------- Barrier Option Classes - Knock-Out ----------------

class DownAndOutCall(Call):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european", rebate: float = 0) -> None:
        """
        Down-and-Out Call : option annulée si le prix passe sous la barrière.
        :param barrier: Niveau de barrière
        :param rebate: Montant versé en cas de knock-out (par défaut 0)
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier
        self.rebate = rebate

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.min(S) <= self.barrier:
            return np.full(1, self.rebate)
        else:
            return np.maximum(S[-1] - self.K, 0)


class DownAndOutPut(Put):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european", rebate: float = 0) -> None:
        """
        Down-and-Out Put : option annulée si le prix passe sous la barrière.
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier
        self.rebate = rebate

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.min(S) <= self.barrier:
            return np.full(1, self.rebate)
        else:
            return np.maximum(self.K - S[-1], 0)


class UpAndOutCall(Call):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european", rebate: float = 0) -> None:
        """
        Up-and-Out Call : option annulée si le prix atteint ou dépasse la barrière.
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier
        self.rebate = rebate

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.max(S) >= self.barrier:
            return np.full(1, self.rebate)
        else:
            return np.maximum(S[-1] - self.K, 0)


class UpAndOutPut(Put):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european", rebate: float = 0) -> None:
        """
        Up-and-Out Put : option annulée si le prix atteint ou dépasse la barrière.
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier
        self.rebate = rebate

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.max(S) >= self.barrier:
            return np.full(1, self.rebate)
        else:
            return np.maximum(self.K - S[-1], 0)


# ---------------- Barrier Option Classes - Knock-In ----------------
class DownAndInCall(Call):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european") -> None:
        """
        Down-and-In Call : l'option s'active uniquement si le prix descend sous la barrière.
        :param barrier: Niveau de barrière
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.min(S) <= self.barrier:
            return np.maximum(S[-1] - self.K, 0)
        else:
            return np.zeros(1)


class DownAndInPut(Put):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european") -> None:
        """
        Down-and-In Put : l'option s'active uniquement si le prix descend sous la barrière.
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.min(S) <= self.barrier:
            return np.maximum(self.K - S[-1], 0)
        else:
            return np.zeros(1)


class UpAndInCall(Call):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european") -> None:
        """
        Up-and-In Call : l'option s'active uniquement si le prix atteint ou dépasse la barrière.
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.max(S) >= self.barrier:
            return np.maximum(S[-1] - self.K, 0)
        else:
            return np.zeros(1)


class UpAndInPut(Put):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european") -> None:
        """
        Up-and-In Put : l'option s'active uniquement si le prix atteint ou dépasse la barrière.
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier

    def payoff(self, S: np.ndarray) -> np.ndarray:
        if np.max(S) >= self.barrier:
            return np.maximum(self.K - S[-1], 0)
        else:
            return np.zeros(1)

# ---------------- Exotic Option Classes ----------------


class AsianCall(Call):
    def __init__(self, K: float, maturity: float, exercise: str = "european") -> None:
        """
        Asian Call : le payoff est basé sur la moyenne des prix de l'actif sur le chemin.
        """
        super().__init__(K, maturity, exercise)

    def payoff(self, S: np.ndarray) -> np.ndarray:
        average_S = np.mean(S)
        return np.maximum(average_S - self.K, 0)


class AsianPut(Put):
    def __init__(self, K: float, maturity: float, exercise: str = "european") -> None:
        """
        Asian Put : le payoff est basé sur la moyenne des prix de l'actif sur le chemin.
        """
        super().__init__(K, maturity, exercise)

    def payoff(self, S: np.ndarray) -> np.ndarray:
        average_S = np.mean(S)
        return np.maximum(self.K - average_S, 0)
