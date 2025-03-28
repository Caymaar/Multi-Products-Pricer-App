import datetime
import numpy as np

class DayCountConvention:
    def __init__(self, convention="Actual/360"):
        """
        Initialise la convention de décompte de jours.

        :param convention: chaîne de caractères indiquant la convention ("Actual/360", "Actual/365", "30/360", "Actual/Actual")
        """
        self.convention = convention.lower()
        self.days_in_year = float(convention.split("/")[0])

    def year_fraction(self, start_date, end_date):
        """
        Calcule la fraction d'année entre deux dates selon la convention choisie.

        :param start_date: date de début (datetime.date ou datetime.datetime)
        :param end_date: date de fin (datetime.date ou datetime.datetime)
        :return: fraction d'année (float)
        """
        if not isinstance(start_date, (datetime.date, datetime.datetime)) or not isinstance(end_date, (
        datetime.date, datetime.datetime)):
            raise TypeError("start_date et end_date doivent être des instances de datetime.date ou datetime.datetime")

        # Différence en nombre de jours réels
        delta_days = (end_date - start_date).days

        if "actual/360" in self.convention:
            return delta_days / 360.0
        elif "actual/365" in self.convention:
            return delta_days / 365.0
        elif "30/360" in self.convention:
            # Utilisation de la convention 30/360 (méthode US)
            d1, m1, y1 = start_date.day, start_date.month, start_date.year
            d2, m2, y2 = end_date.day, end_date.month, end_date.year
            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 == 30:
                d2 = 30
            return ((360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0)
        elif "actual/actual" in self.convention:
            # Méthode simplifiée : en cas de traversée de plusieurs années, on peut raffiner.
            # Ici, on utilise une moyenne approximative de 365.25 jours par an.
            return delta_days / 365.25
        else:
            raise ValueError(f"Convention inconnue : {self.convention}")


# Exemple d'utilisation dans un contexte de pricing
if __name__ == "__main__":
    # Dates d'exemple
    today = datetime.date(2025, 3, 26)
    future_date = datetime.date(2026, 3, 26)

    # Création d'une instance pour chaque convention
    convention_actual360 = DayCountConvention("Actual/360")
    convention_actual365 = DayCountConvention("Actual/365")
    convention_30_360 = DayCountConvention("30/360")
    convention_actual_actual = DayCountConvention("Actual/Actual")

    # Calcul de la fraction d'année entre today et future_date
    print("Actual/360:", convention_actual360.year_fraction(today, future_date))
    print("Actual/365:", convention_actual365.year_fraction(today, future_date))
    print("30/360:", convention_30_360.year_fraction(today, future_date))
    print("Actual/Actual:", convention_actual_actual.year_fraction(today, future_date))

    # Exemple dans le pricing d'une option :
    # Supposons que nous avons un taux annuel r et que nous souhaitons actualiser un cash flow
    r = 0.05  # 5% de taux annuel
    T = convention_actual360.year_fraction(today, future_date)  # Temps en années selon la convention Actual/360
    discount_factor = np.exp(-r * T)
    print("Discount factor (Actual/360):", discount_factor)
