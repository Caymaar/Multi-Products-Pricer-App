import numpy as np


def bootstrap_zero_curve(taus: np.ndarray, swap_rates: np.ndarray, freq: int = 1):
    """
    Bootstrapping d'une courbe zéro-coupon à partir de swap par rates.

    :param tenors: liste de maturités ['1Y','2Y',...]
    :param swap_rates: array de par rates (en décimal, p.ex. 0.02 pour 2%)
    :param freq: fréquence des paiements fixes par an (1=annuel, 2=semi-annuel...)
    :return: tuple (discount_factors, zero_rates)
    """

    # Calcule des year fractions entre paiements
    # dt_i = tau_i - tau_{i-1}, avec tau_0 = 0
    year_fracs = np.diff(np.concatenate(([0.0], taus)))

    n = len(taus)
    dfs = np.zeros(n)

    # Bootstrapping itératif
    for i in range(n):
        c = swap_rates[i]
        dt = year_fracs[i]

        if i == 0:
            # Premier point : simple discount
            dfs[i] = 1.0 / (1.0 + c * dt)
        else:
            # Somme des flux fixes actualisés jusqu'à i-1
            fixed_pv = sum(c * year_fracs[j] * dfs[j] for j in range(i))
            # Equation par rate : c * sum_{j=1}^i DF_j * dt_j + DF_i = 1
            dfs[i] = (1.0 - fixed_pv) / (1.0 + c * dt)

    # Taux zéro-coupon
    zero_rates = -np.log(dfs) / taus

    return zero_rates