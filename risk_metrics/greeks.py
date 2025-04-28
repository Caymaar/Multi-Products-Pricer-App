from option.option import OptionPortfolio
from pricers.mc_pricer import MonteCarloEngine
from pricers.tree_pricer import TreeModel, TreePortfolio
from datetime import timedelta
import copy
from typing import Union
import numpy as np


class GreeksCalculator:
    def __init__(self, model, epsilon=1e-2, type="MC"):
        """
        Initialise le calculateur de Greeks avec un modèle MC.

        :param model: Instance du modèle.
        :param epsilon: Pas de variation pour les dérivées numériques.
        :param type: Méthode de pricing ("Longstaff / MC / Trinomial").
        """
        self.epsilon = epsilon
        self._type = type
        self._original_model = copy.copy(model)  # Sauvegarde de l'état initial
        self._cached_prices = {}  # Stocke les prix calculés pour éviter les répétitions
        self._alphas = self._original_model._alpha if hasattr(self._original_model, "_alpha") else None

    def _recreate_model(self, **kwargs):
        new_params = {
            "market": copy.deepcopy(self._original_model.market),
            "option_ptf": copy.deepcopy(self._original_model.options),
            "pricing_date": self._original_model.pricing_date,
            "n_steps": self._original_model.n_steps,
        }

        if issubclass(type(self._original_model), MonteCarloEngine):
            proc = next(iter(self._original_model.diffusions.values()))
            new_params["n_paths"] = self._original_model.n_paths
            new_params["seed"] = proc.brownian.seed

        new_params.update(kwargs)
        return type(self._original_model)(**new_params)

    def _get_price(self, model, key, up=False, down=False):
        if key not in self._cached_prices:
            if isinstance(model, Union[TreeModel, TreePortfolio]):
                self._cached_prices[key] = model.price(up=up, down=down)
            else:
                self._cached_prices[key] = model.price(self._type)
        return self._cached_prices[key]

    def delta(self):
        """Calcul optimisé du Delta : dPrix/dS0 selon le modèle"""
        S0 = self._original_model.market.S0
        key_up, key_down = "S0_up", "S0_down"

        if isinstance(self._original_model, Union[TreeModel, TreePortfolio]):
            price_up = self._get_price(self._original_model, key_up, up=True)
            price_down = self._get_price(self._original_model, key_down, down=True)
            return (price_up - price_down) / (S0 * self._alphas - S0 / self._alphas)

        mc_up = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 + self.epsilon)))
        price_up = self._get_price(mc_up, key_up, up=True)

        mc_down = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 - self.epsilon)))
        price_down = self._get_price(mc_down, key_down, down=True)

        return (price_up - price_down) / (2 * S0 * self.epsilon)

    def gamma(self):
        """Calcul optimisé du Gamma : d²Prix/dS0²"""
        S0 = self._original_model.market.S0
        key_up, key_down, key_mid = "S0_up", "S0_down", "S0"

        if isinstance(self._original_model, Union[TreeModel, TreePortfolio]):
            model_up, model_down = self._original_model, self._original_model
        else:
            model_up = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 + self.epsilon)))
            model_down = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 - self.epsilon)))

        price = self._get_price(self._original_model, key_mid)
        price_up = self._get_price(model_up, key_up, up=True)
        price_down = self._get_price(model_down, key_down, down=True)

        if isinstance(self._original_model, Union[TreeModel, TreePortfolio]):
            d_up = (price_up - price) / (self._alphas * S0 - S0)
            d_down = (price - price_down) / (S0 - S0 / self._alphas)
            return (d_up - d_down) / ((self._alphas * S0 - S0 / self._alphas) / 2)

        return (price_up - 2 * price + price_down) / (S0 ** 2 * self.epsilon ** 2)

    def vega(self):
        """Calcul du Vega : dPrix/dSigma"""
        sigma = self._original_model.market.sigma

        model_up = self._recreate_model(market=self._original_model.market.copy(sigma=sigma + self.epsilon))
        model_down = self._recreate_model(market=self._original_model.market.copy(sigma=sigma - self.epsilon))

        price_up = self._get_price(model_up, "sigma_up")
        price_down = self._get_price(model_down, "sigma_down")

        return (price_up - price_down) / (2 * self.epsilon) / 100

    def theta(self):
        """Calcul du Theta : dPrix/dt"""
        pricing_date = self._original_model.pricing_date + timedelta(days=1)

        model_new = self._recreate_model(pricing_date=pricing_date)
        price_new = self._get_price(model_new, "t_future")
        price_old = self._get_price(self._original_model, "t_now")

        return price_new - price_old

    def rho(self):
        """Calcul du Rho : dPrix/dr"""
        r = self._original_model.market.r

        model_up = self._recreate_model(market=self._original_model.market.copy(r=r + self.epsilon))
        model_down = self._recreate_model(market=self._original_model.market.copy(r=r - self.epsilon))

        price_up = self._get_price(model_up, "r_up")
        price_down = self._get_price(model_down, "r_down")

        return (price_up - price_down) / (2 * self.epsilon) / 100

    def speed(self):
        """
        Calcul du Speed : d³Prix/dS0³
        """
        S0 = self._original_model.market.S0
        epsilon = self.epsilon

        if isinstance(self._original_model, TreePortfolio):
            speeds = []

            for tree, options in self._original_model.trees.values():
                alpha = tree.alpha  # alpha propre à cette option

                for option in options:
                    idx = self._original_model.options.assets.index(option)  # Retrouver l'indice de l'option et son poids
                    weight = self._original_model.options.weights[idx]
                    opt_ptf = OptionPortfolio([option],[weight])

                    S_up = S0 * alpha
                    S_up2 = S0 * alpha**2
                    S_down = S0 / alpha
                    S_down2 = S0 / alpha**2

                    # Recrée des petits TreeModel individuels
                    tree_up, tree_down = self._original_model.recreate_model(option_ptf=opt_ptf), self._original_model.recreate_model(option_ptf=opt_ptf)
                    tree_up2 = self._original_model.recreate_model(option_ptf=opt_ptf, market=tree.market.copy(S0=S_up))
                    tree_down2 = self._original_model.recreate_model(option_ptf=opt_ptf, market=tree.market.copy(S0=S_down))

                    # Calcule prix pour l'option seule
                    price_up2 = tree_up2.price(up=True)
                    price_up = tree_up.price(up=True)
                    price_down2 = tree_down2.price(down=True)
                    price_down = tree_down.price(down=True)

                    g_up = (price_up2 - price_up) / (S_up2 - S_up)
                    g_down = (price_down - price_down2) / (S_down - S_down2)

                    speed_i = (g_up - g_down) / ((S_up + S_down) / 2)
                    speeds.append(speed_i)

            return np.array(speeds)  # Vecteur de speeds par option

        else:
            model_up = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 + self.epsilon)))
            model_down = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 - self.epsilon)))
            model_up2 = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 + 2 * epsilon)))
            model_down2 = self._recreate_model(market=self._original_model.market.copy(S0=S0 * (1 - 2 * epsilon)))

            # Calcul des prix correspondants
            price_up2 = self._get_price(model_up2, "S0_up2", up=True)
            price_up = self._get_price(model_up, "S0_up", up=True)
            price_down = self._get_price(model_down, "S0_down", down=True)
            price_down2 = self._get_price(model_down2, "S0_down2", down=True)

            numerator = price_up2 - 2 * price_up + 2 * price_down - price_down2
            denominator = 2 * (epsilon * S0) ** 3

            return numerator / denominator

    def all_greeks(self):
        return [self.delta(), self.gamma(), self.vega(), self.theta(), self.rho(), self.speed()]
