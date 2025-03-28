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
