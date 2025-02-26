import numpy as np
import pandas as pd 
from typing import Dict 
from .Technical_Indicator import Technical_Indicator

class AlgorithmicSignals:
    """Generate trading signals based on technical analysis"""
    
    def __init__(self):
        self.indicators = Technical_Indicator()
        
    def generate_signals(self, data: pd.DataFrame) -> Dict[str, np.array]:
        """
        Generate trading signals based on multiple technical indicators
        
        Args:
            df: DataFrame with 'close' and 'volume' columns
            
        Returns:
            Dictionary of signal arrays for different strategies
        """
        signals = {}
    
        # RSI signal
        rsi = self.indicators.calculate_rsi(data['close'])
        signals['rsi_signal'] = (rsi > 30).astype(int)  #Oversold condition 
        
        # MACD signal
        macd, signal, _ = self.indicators.calculate_macd(data['close'])
        signals['macd_signal'] = ((macd > signal) & (macd < 0)).astype(int)
    
        # Bollinger Bands signal
        upper, middle, lower = self.indicators.bolloinger_bands(data['close'])
        signals['bb_signal'] = (data['close'] < lower).astype(int)
        
        # Volume indicators signal
        volume_indicators = self.indicators.calculate_volume_indicators(data['close'], 
                                                                        data['volume'])
        signals['volume_indicators'] = (data['volume'] > volume_indicators['volume_sma'] * 
                                        1.5).astype(int)
        
        return signals
    def combine_signals(self, signals: Dict[str, np.array]) -> np.array:
        """
        Combine multiple signals into a single trading signal
        
        Args:
            signals: Dictionary of signal arrays for different strategies
            
            weights: Optional dictionary of weights for each signal
            
        Returns:
            Combined trading signal
        """
        if weights is None:
            weights = {
                'rsi_signal': 0.3,
                'macd_signal': 0.3,
                'bb_signal': 0.2,
                'volume_indicators': 0.2
            }
            
            # Combine weighted signals
        combined = np.zeros_like(signals['rsi_signal'], dtype=float)
        for signal_type, signal in signals.items():
            combined += weights[signal_type] * signal
            
        return (combined > 0.5).astype(int)