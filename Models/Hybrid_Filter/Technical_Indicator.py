import numpy as np
import pandas as pd 
from typing import Dict

class Technical_Indicator:
    """Technical analysis indicators calculator"""
    
    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(data: pd.Series)-> tuple:
        """ Calculate MACD, Signal line and Histogram"""
        exp1 = data.ewm(span=12, adjust=False).mean()
        exp2 = data.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    @staticmethod
    def bolloinger_bands(data: pd.Series, window: int = 20, num_std: int = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = data.rolling(window=window).mean()
        rolling_std = data.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return {'upper_band': upper_band, 'lower_band': lower_band}
    
    @staticmethod
    def calculate_volume_indicators(price: pd.Series, volume: pd.Series)-> Dict[str, pd.Series]:
        """Calculate volume-base indicators"""
        # volume moving average
        volume_sma = volume.rolling(window=20).mean()
        
        #on balance volume
        obv = (volume * (~price.diff().le(0) * 2 - 1)).cumsum()
        
        # volume price trend
        vpt = volume * price.pct_change()
        vpt = vpt.cumsum()
        
        return{
            'volume_sma': volume_sma,
            'obv': obv,
            'vpt': vpt
        }
        
       
    