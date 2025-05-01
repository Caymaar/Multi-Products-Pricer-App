from typing import Any
from risk_metrics.sensitivity import SensitivityAnalyzer

class SensitivityFactory:
    """
    Usine qui crée automatiquement l’analyzer adapté
    à une Strategy (vanille ou structuré) en inférant
    le modèle de pricing (MC, Trinomial ou Black‐Scholes).
    """

    @staticmethod
    def create_from_strategy(
        strategy: Any,
        market: Any,
        pricing_date: Any,
        *,
        method: str = "MC",
        n_paths: int = 10_000,
        n_steps: int = 300,
        seed: int | None = None,
        compute_antithetic: bool = False,
        epsilon: float = 1e-4
    ) -> SensitivityAnalyzer:
        """
        Construit l’engine à partir des legs de la stratégie puis
        renvoie un SensitivityAnalyzer.

        :param strategy: instance de Strategy (get_legs(), price(engine))
        :param market:    instance de Market
        :param pricing_date: date de valorisation
        :param method:    "MC", "Longstaff", "Trinomial" ou "BS"
        :param n_paths:   pour MC
        :param n_steps:   pour MC / Trinomial
        :param seed:      pour MC
        :param compute_antithetic: pour MC
        :param epsilon:   bump pour dérivées numériques
        """
        from option.option import OptionPortfolio, Option
        from pricers.mc_pricer  import MonteCarloEngine
        from pricers.tree_pricer import TreePortfolio
        from pricers.bs_pricer   import BSPortfolio

        # 1) on ne garde que les jambes d’option
        legs = strategy.get_legs()
        opt_legs = [(leg, w) for leg, w in legs if isinstance(leg, Option)]
        if not opt_legs:
            raise ValueError("La stratégie ne contient pas de legs d’option pour ce factory.")

        opts, weights = zip(*opt_legs)
        ptf = OptionPortfolio(list(opts), list(weights))

        # 2) on instancie l’engine qui convient
        method = method.lower()
        if method in ("mc", "longstaff"):
            engine = MonteCarloEngine(
                market=market,
                option_ptf=ptf,
                pricing_date=pricing_date,
                n_paths=n_paths,
                n_steps=n_steps,
                seed=seed,
                compute_antithetic=compute_antithetic
            )
        elif method == "trinomial":
            engine = TreePortfolio(
                market=market,
                option_ptf=ptf,
                pricing_date=pricing_date,
                n_steps=n_steps
            )
        elif method == "bs":
            engine = BSPortfolio(
                market=market,
                option_ptf=ptf,
                pricing_date=pricing_date
            )
        else:
            raise ValueError(f"Méthode inconnue pour le factory: {method}")

        # 3) on crée et renvoie l’analyzer
        return SensitivityAnalyzer(
            strategy=strategy,
            engine=engine,
            epsilon=epsilon,
            method=method.upper()
        )
