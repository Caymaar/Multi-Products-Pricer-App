from datetime import datetime
import numpy as np

class DayCountConvention:
    def __init__(self, convention="Actual/365"):
        """
        Initialise la convention de décompte de jours.

        :param convention: chaîne de caractères indiquant la convention ("Actual/360", "Actual/365", "30/360", "Actual/Actual")
        """
        conv = convention.lower()
        self.convention = conv

        parts = conv.split("/")
        if len(parts) != 2:
            raise ValueError(f"Convention invalide : {convention!r}")
        num, den = parts
        self.num = num
        self.den = den

        # jours/an pour actual/xxx
        if num == "actual" and den in ("360", "365"):
            self.days_in_year = float(den)
        else:
            # on n'utilisera pas days_in_year pour 30/360 ni Actual/Actual
            self.days_in_year = None

    def year_fraction(self, start_date, end_date):
        """
        Calcule la fraction d'année entre deux dates selon la convention choisie.

        :param start_date: date de début (datetime.date ou datetime.datetime)
        :param end_date: date de fin (datetime.date ou datetime.datetime)
        :return: fraction d'année (float)
        """


        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            try:
                start_date = datetime.combine(start_date, datetime.min.time())
                end_date = datetime.combine(end_date, datetime.min.time())
            except TypeError:
                raise TypeError(f"start_date et end_date doivent être des instances de datetime.date ou datetime.datetime. start_date: {start_date}, end_date: {end_date}")

        delta_days = (end_date - start_date).days

        try:
            # --- Actual/365 or Actual/360 ---
            if self.num == "actual" and self.den in ("360", "365"):
                return delta_days / self.days_in_year

            # --- 30/360 US ---
            if self.num == "30" and self.den == "360":
                y1, m1, d1 = start_date.year, start_date.month, start_date.day
                y2, m2, d2 = end_date.year,   end_date.month,   end_date.day
                # ajustements US
                if d1 == 31:
                    d1 = 30
                if d2 == 31 and d1 == 30:
                    d2 = 30
                days_30_360 = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
                return days_30_360 / 360.0

            # --- Actual/Actual ---
            if self.num == "actual" and self.den == "actual":
                total = 0.0
                current = start_date
                while current < end_date:
                    # fin de l'année de cursor
                    year_end = datetime(current.year + 1, 1, 1)
                    sub_end  = min(end_date, year_end)
                    days_in_year = (year_end - datetime(current.year, 1, 1)).days
                    total += (sub_end - current).days / days_in_year
                    current = sub_end
                return total
        except:
            raise ValueError(f"Convention non supportée : {self.convention!r}")


# Exemple d'utilisation dans un contexte de pricing
if __name__ == "__main__":
    # Dates d'exemple
    today = datetime(2025, 3, 26)
    future_date = datetime(2026, 3, 26)

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
