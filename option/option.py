import numpy as np
from abc import ABC, abstractmethod

# ---------------- Base Option Class ----------------
class Option(ABC):
    def __init__(self, K, maturity, exercise = "european"):
        self.K = K  # Strike price
        self.T = maturity  # Time to maturity
        self.exercise = exercise.lower() # European / American / Other

    @abstractmethod
    def payoff(self, S):
        """ Méthode de polymorphisme abstraite pour calculer le payoff d'une option. """
        raise NotImplementedError("La méthode payoff doit être implémentée dans les sous-classes.")

# ---------------- Call and Put Option Classes ----------------
class Call(Option):
    def __init__(self, K, maturity, exercise):
        Option.__init__(self, K, maturity, exercise)

    def payoff(self, S):
        """ Payoff d'un Call à maturité. """
        return np.maximum(S - self.K, 0)

class Put(Option):
    def __init__(self, K, maturity, exercise):
        Option.__init__(self, K, maturity, exercise)

    def payoff(self, S):
        """ Payoff d'un Put à maturité. """
        return np.maximum(self.K - S, 0)


# ---------------- Barrier Option Classes ----------------

class DownAndOutCall(Call):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european", rebate: float = 0) -> None:
        """
        Down-and-Out Call : option annulée si le prix passe en dessous du niveau barrier.
        :param barrier: Niveau de barrière
        :param rebate: Montant versé en cas de knock-out
        """
        super().__init__(K, maturity, exercise)
        self.barrier = barrier
        self.rebate = rebate

    def payoff(self, S: np.ndarray) -> np.ndarray:
        """
        Si le chemin de prix S atteint ou descend sous la barrière, l'option est knockout (payoff = rebate).
        Sinon, le payoff est celui d'un Call classique.
        """
        if np.min(S) <= self.barrier:
            return np.full(1, self.rebate)
        else:
            return np.maximum(S[-1] - self.K, 0)


class DownAndOutPut(Put):
    def __init__(self, K: float, maturity: float, barrier: float, exercise: str = "european", rebate: float = 0) -> None:
        """
        Down-and-Out Put : option annulée si le prix descend sous la barrière.
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