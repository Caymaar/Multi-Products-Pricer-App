import sys
import os

# Ajouter dynamiquement le chemin du répertoire principal du projet
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from data.management.data_utils import get_price_data, get_rate_data, get_implied_vol, get_zone
import numpy as np
from datetime import datetime
import pandas as pd

RATE_DICT = {
    "RFR": {"EU": "ESTER", "US": "SOFR"},
    "FORWARD": {"EU": "EURIBOR"}
}

class DataRetriever:
    def __init__(self, stock_name:str):
        self.stock_name = stock_name
        zone = get_zone(stock_name)
        self.prices = get_price_data(stock_name)

        self.risk_free_index = get_rate_data(RATE_DICT['RFR'][zone], "SPOT")
        self.risk_free_curve = get_rate_data(RATE_DICT['RFR'][zone], "CURVE")
        self.floating_index = get_rate_data(RATE_DICT['FORWARD'][zone], "SPOT")
        self.floating_curve = get_rate_data(RATE_DICT['FORWARD'][zone], "CURVE")
        try:
            self.implied = get_implied_vol(stock_name)
        except Exception as e:
            print(f"Error retrieving implied volatility: {e}")
            self.implied = None

    def get_prices(self, date: datetime):
        return self.prices.iloc[self.prices.index <= date].iloc[-1]
    
    def get_risk_free_index(self, date: datetime):
        return self.risk_free_index.iloc[self.risk_free_index.index <= date].iloc[-1]
    
    def get_risk_free_curve(self, date: datetime):
        return self.risk_free_curve.iloc[self.risk_free_curve.index <= date].iloc[-1] / 100

    def get_floating_index(self, date: datetime):
        return self.floating_index.iloc[self.floating_index.index <= date].iloc[-1]

    def get_floating_curve(self, date: datetime):
        return self.floating_curve.iloc[self.floating_curve.index <= date].iloc[-1] / 100

    def get_date_range(self):
        prices_range = (self.prices.index.min(), self.prices.index.max())
        spot_range = (self.risk_free_index.index.min(), self.risk_free_index.index.max())
        curve_range = (self.risk_free_curve.index.min(), self.risk_free_curve.index.max())
        
        common_min = max(prices_range[0], spot_range[0], curve_range[0])
        common_max = min(prices_range[1], spot_range[1], curve_range[1])
        
        return common_min, common_max
    
    def get_implied_volatility(self, date: datetime = None):
        return self.implied
    
    def get_correlation(self, date: datetime, maturity: datetime):
        if maturity < date:
            raise ValueError("Maturity date must be after the current date.")
        
        delta = (maturity - date).days

        start_date = date - pd.DateOffset(days=2*delta)

        risk_free_index = self.risk_free_index.loc[start_date:date]
        prices = np.log(self.prices.loc[start_date:date])

        merged_df = pd.merge(risk_free_index, prices, left_index=True, right_index=True)
        merged_df.columns = ['RFR', 'Prices']
        merged_df = merged_df.diff()
        merged_df = merged_df.dropna()

        return merged_df.corr()

    def get_historical_volatility(self, date: datetime, window: int = 252) -> float:
        # log‐returns journaliers
        rets = np.log(self.prices / self.prices.shift(1)).dropna()
        # volatilité annualisée sur la fenêtre précédente
        hist = rets.loc[:date].tail(window).std() * np.sqrt(window)
        return hist


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    stock_name = "AMAZON"
    retriever = DataRetriever(stock_name)

    # On prend une date dans l'intervalle commun
    start_date, end_date = retriever.get_date_range()
    sample_date = start_date + (end_date - start_date) / 2
    sample_date = pd.to_datetime(sample_date).normalize()  # s'assurer que c'est bien un datetime sans heure

    # Récupération du taux spot
    spot = retriever.get_risk_free_index(sample_date)
    print(f"Risk-free SPOT rate on {sample_date.date()}: {spot:.4%}")

    # Récupération de la courbe de taux
    curve = retriever.get_risk_free_curve(sample_date)
    print(f"\nRisk-free CURVE on {sample_date.date()}:")
    print(curve)

    # Optionnel : si tu veux visualiser la courbe de taux
    if isinstance(curve, pd.Series):
        plt.figure(figsize=(10, 5))
        tenors = curve.index
        rates = curve.values
        plt.plot(tenors, rates, marker='o')
        plt.title(f"Risk-Free Curve on {sample_date.date()} for {stock_name}")
        plt.xlabel("Tenor")
        plt.ylabel("Rate")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()