import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

# Standard library imports
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.preprocessing import MinMaxScaler

# Try relative import first (works when run as module)
# If it fails, fall back to absolute import (works when run directly)
try:
    from .Algorithmic_signals import AlgorithmicSignals
except ImportError:
    from Models.Hybrid_Filter.Algorithmic_signals import AlgorithmicSignals

# Other imports
from Models.Bitcoin.BitC_Predictor import BitC_Predictor
from Models.Ethereum.Ethereum_Predictor import Ethereum_Predictor
from Models.Tether.Tether_Predictor import Tether_Prediction
from Models.USDCoin.usdC__Predictor import usdC__Predictor

class HybridPredictor:
    """
    Combines ML predictions with algorithmic trading signals
    """
    def __init__(self, look_back: int = 60):
        self.look_back = look_back
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Initialize predictors
        self.predictors = {
            'BTC': BitC_Predictor(look_back=look_back),
            'ETH': Ethereum_Predictor(look_back=look_back),
            'USDT': Tether_Prediction(look_back=look_back),
            'USDC': usdC__Predictor(look_back=look_back)
        }
        
        # Initialize algorithmic signal generator
        self.algo_signals = AlgorithmicSignals()
        
    def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """
        Generate predictions and trading signals
        
        Args:
            data: Dictionary of DataFrames for each cryptocurrency
            
        Returns:
            Dictionary containing predictions and signals for each crypto
        """
        results = {}
        
        for crypto, df in data.items():
            if crypto not in self.predictors:
                continue
                
            # Get ML predictions
            ml_predictions = self.predictors[crypto].predict(df)
            
            # Get algorithmic signals
            algo_signals = self.algo_signals.generate_signals(df)
            combined_algo_signal = self.algo_signals.combine_signals(algo_signals)
            
            # Combine ML and algorithmic approaches
            final_signals = self.combine_predictions(
                ml_predictions,
                combined_algo_signal
            )
            
            results[crypto] = {
                'prediction': ml_predictions,
                'algo_signal': combined_algo_signal,
                'final_signal': final_signals,
                'technical_signals': algo_signals
            }
            
        return results
        
    def combine_predictions(self, 
                          ml_pred: np.ndarray, 
                          algo_signal: np.ndarray,
                          ml_weight: float = 0.7) -> np.ndarray:
        """
        Combine ML predictions with algorithmic signals
        
        Args:
            ml_pred: ML model predictions
            algo_signal: Algorithmic trading signals
            ml_weight: Weight for ML predictions (0-1)
            
        Returns:
            Array of final trading signals
        """
        # Scale ML predictions to 0-1 range
        scaled_pred = self.scaler.fit_transform(ml_pred.reshape(-1, 1)).flatten()
        
        # Combine signals
        combined = (ml_weight * scaled_pred + 
                   (1 - ml_weight) * algo_signal)
        
        return (combined > 0.5).astype(int)
        
    def analyze_signals(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """
        Analyze and summarize trading signals
        
        Args:
            results: Dictionary of prediction results
            
        Returns:
            DataFrame with signal analysis
        """
        analysis = []
        
        for crypto, data in results.items():
            # Calculate signal agreement
            signal_agreement = np.mean(
                data['algo_signal'] == data['final_signal']
            )
            
            # Calculate signal transitions
            transitions = np.sum(np.diff(data['final_signal']) != 0)
            
            analysis.append({
                'crypto': crypto,
                'signal_agreement': signal_agreement,
                'signal_transitions': transitions,
                'buy_signals': np.sum(data['final_signal']),
                'total_signals': len(data['final_signal'])
            })
            
        return pd.DataFrame(analysis)
        
    def generate_report(self, results: Dict[str, Dict]) -> str:
        """
        Generate a human-readable report of predictions and signals
        
        Args:
            results: Dictionary of prediction results
            
        Returns:
            Formatted report string
        """
        report = "Hybrid Prediction Report:\n\n"
        
        for crypto, data in results.items():
            report += f"{crypto} Analysis:\n"
            report += f"Latest Price Prediction: ${data['prediction'][-1]:.2f}\n"
            report += f"Trading Signal: {'BUY' if data['final_signal'][-1] else 'HOLD/SELL'}\n"
            
            # Add technical analysis summary
            tech_signals = data['technical_signals']
            report += "Technical Indicators:\n"
            report += f"- RSI Signal: {'Oversold' if tech_signals['rsi_signal'][-1] else 'Normal'}\n"
            report += f"- MACD Signal: {'Buy' if tech_signals['macd_signal'][-1] else 'Hold/Sell'}\n"
            report += f"- Volume Signal: {'High' if tech_signals['volume_signal'][-1] else 'Normal'}\n"
            report += "\n"
            
        return report