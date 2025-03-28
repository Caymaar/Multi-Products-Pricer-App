import numpy as np

class Regression:

    @staticmethod
    def fit(reg_type, X, Y):
        if len(X) == 0:
            return np.zeros_like(Y)

        # Mapping entre le nom et l'ordre du polynôme
        reg_types = {
            "Linear": 1,
            "Quadratic": 2,
            "Cubic": 3,
            "Quartic": 4,
            "Quintic": 5,
            "Sextic": 6
        }

        if reg_type not in reg_types:
            raise ValueError(f"Type de régression '{reg_type}' non reconnu. "
                             f"Choisissez parmi {list(reg_types.keys())}.")

        degree = reg_types[reg_type]  # Récupération du degré correspondant

        # Construction de la matrice X_reg avec les puissances de X jusqu'à 'degree'
        X_reg = np.vstack([X ** i for i in range(degree + 1)]).T

        # Régression par moindres carrés
        coeffs, _, _, _ = np.linalg.lstsq(X_reg, Y, rcond=None)

        # Calcul des valeurs prédites
        continuation = X_reg @ coeffs  # Produit matriciel

        return continuation