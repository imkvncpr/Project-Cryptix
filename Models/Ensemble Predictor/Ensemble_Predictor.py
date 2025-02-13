import numpy as np
import pandas as pd
import tensorflow as tf
Model = tf.keras.models.Model
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM
Concatenate = tf.keras.layers.Concatenate
Input = tf.keras.layers.Input
Dropout = tf.keras.layers.Dropout
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import plotly.graph_objects as go
import os
import sys
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


from Models.Bitcoin.BitC_Predictor import BitC_Predictor  # This one looks correct
from Models.Ethereum.Ethereum_Predictor import Ethereum_Predictor  # You'll need to create this file - I see best_model.h5 instead
from Models.Tether.Tether_Predictor import Tether_Prediction  # Note: file is Tether_Prediction.py
from Models.USDCoin.usdC__Predictor import usdC__Predictor  # Note: file is USDC_Predictor.py

class EnsemblePredictor:
    def __init__(self, look_back: int = 60, test_size: float = 0.2):
        """Initialize the ensemble predictor."""
        self.look_back = look_back
        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {
            'BTC': BitC_Predictor(look_back=look_back, test_size=test_size),
            'ETH': Ethereum_Predictor(look_back=look_back, test_size=test_size),
            'USDT': Tether_Prediction(look_back=look_back, test_size=test_size),
            'USDC': usdC__Predictor(look_back=look_back, test_size=test_size)
        }
        self.model = None
        
    def get_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[List[np.ndarray], np.ndarray]:
        """Prepare data for training the ensemble model."""
        X_sequences = []
        y_values = []
        
        for coin, df in data.items():
            if coin not in self.models:
                continue
            X, y = self.models[coin].prepare_data(df)
            X_sequences.append(X)
            y_values.append(y)
            
        if not X_sequences:
            raise ValueError("No valid data provided")
            
        # Use mean of all predictions as target
        y_combined = np.mean(y_values, axis=0)
        
        # Fit the scaler during training
        y_combined = self.scaler.fit_transform(y_combined.reshape(-1, 1)).flatten()
        
        return X_sequences, y_combined
    
    def create_model(self) -> tf.keras.models.Model:
        """Create the ensemble model architecture."""
        if any(model.model is None for model in self.models.values()):
            raise ValueError("All individual models must be created first")
            
        inputs = []
        outputs = []
        for model in self.models.values():
            input_layer = Input(shape=model.model.input_shape[1:])
            inputs.append(input_layer)
            # Use model layers up to the second-to-last layer
            x = model.model.layers[0](input_layer)
            for layer in model.model.layers[1:-1]:
                x = layer(x)
            outputs.append(x)

        merged = Concatenate()(outputs)
        merged = Dense(50, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = Dense(25, activation='relu')(merged)
        merged = Dense(1)(merged)
        
        model = Model(inputs=inputs, outputs=merged)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def fit(self, data: Dict[str, pd.DataFrame], epochs: int = 100, batch_size: int = 32):
        """Train both individual models and ensemble model."""
        # First train individual models
        for coin, model in self.models.items():
            if coin in data:
                print(f"\nTraining {coin} model...")
                model.fit(data[coin], epochs=epochs, batch_size=batch_size)
        
        # Then train ensemble model
        print("\nTraining ensemble model...")
        X_sequences, y = self.get_data(data)
        self.model = self.create_model()
        return self.model.fit(X_sequences, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
            
    def predict(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, np.ndarray]]:
        """Make predictions using both individual and ensemble models."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        X_sequences, _ = self.get_data(data)
        ensemble_preds = self.model.predict(X_sequences)
        
        for coin, model in self.models.items():
            if coin in data:
                ind_preds = model.predict(data[coin])
                predictions[coin] = {
                    'individual': ind_preds,
                    'ensemble': self.scaler.inverse_transform(ensemble_preds.reshape(-1, 1)).flatten()
                }
        
        return predictions

    def predict_future(self, data: Dict[str, pd.DataFrame], days: int = 30) -> Dict[str, Dict[str, List[float]]]:
        """Predict future prices for specified number of days."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = {}
        
        # Get last sequence for each model
        sequences = {}
        for coin, model in self.models.items():
            if coin in data:
                sequences[coin] = model.get_last_sequence(data[coin])
        
        # Make predictions for each day
        for coin, model in self.models.items():
            if coin not in data:
                continue
                
            ind_preds = []
            ens_preds = []
            current_sequence = sequences[coin].copy()
            
            for _ in range(days):
                # Get individual prediction
                ind_pred = model.model.predict(current_sequence)[0, 0]
                ind_preds.append(float(ind_pred))
                
                # Get ensemble prediction
                ens_pred = self.model.predict([current_sequence])[0, 0]
                ens_preds.append(float(ens_pred))
                
                # Update sequence for next prediction
                new_sequence = np.roll(current_sequence[0], -1, axis=0)
                new_sequence[-1] = ind_pred
                current_sequence = new_sequence.reshape(1, *current_sequence.shape[1:])
            
            predictions[coin] = {
                'individual': ind_preds,
                'ensemble': ens_preds
            }
        
        return predictions
    
    def plot_predictions(self, actual_data: Dict[str, pd.DataFrame], predictions: Dict[str, Dict[str, np.ndarray]]):
        """Plot actual and predicted prices."""
        fig = go.Figure()
        
        for coin in self.models.keys():
            if coin not in actual_data or coin not in predictions:
                continue
            
            # Plot actual data
            fig.add_trace(go.Scatter(
                x=actual_data[coin].index,
                y=actual_data[coin]['Close'],
                mode='lines',
                name=f'{coin} Actual'
            ))
            
            # Plot individual model predictions
            fig.add_trace(go.Scatter(
                x=actual_data[coin].index[-len(predictions[coin]['individual']):],
                y=predictions[coin]['individual'],
                mode='lines',
                name=f'{coin} Individual',
                line=dict(dash='dash')
            ))
            
            # Plot ensemble predictions
            fig.add_trace(go.Scatter(
                x=actual_data[coin].index[-len(predictions[coin]['ensemble']):],
                y=predictions[coin]['ensemble'],
                mode='lines',
                name=f'{coin} Ensemble',
                line=dict(dash='dot')
            ))
        
        fig.update_layout(
            title='Cryptocurrency Price Predictions',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )
        
        fig.show()
        
    def save_models(self, path: str = 'Saved_Models'):
        """Save all models."""
        os.makedirs(path, exist_ok=True)
        
        # Save individual models
        for coin, model in self.models.items():
            model.save_model(os.path.join(path, f'best_model_{coin}.h5'))
        
        # Save ensemble model
        if self.model:
            self.model.save(os.path.join(path, 'best_model.h5'))
    
    def load_models(self, path: str = 'Saved_Models'):
        """Load all models."""
        # Load individual models
        for coin, model in self.models.items():
            model_path = os.path.join(path, f'best_model_{coin}.h5')
            if os.path.exists(model_path):
                model.load_model(model_path)
        
        # Load ensemble model
        ensemble_path = os.path.join(path, 'best_model.h5')
        if os.path.exists(ensemble_path):
            self.model = tf.keras.models.load_model(ensemble_path)

if __name__ == '__main__':
    # Example usage
    predictor = EnsemblePredictor()
    
    # Download and prepare data
    data = {}
    for coin, model in predictor.models.items():
        data[coin] = model.download_data(years=2)
    
    if all(v is not None for v in data.values()):
        # Train models
        history = predictor.fit(data)
        
        # Make predictions
        predictions = predictor.predict(data)
        
        # Plot results
        predictor.plot_predictions(data, predictions)
        
        # Save models
        predictor.save_models()
        print("\nModels trained and saved successfully")
    else:
        print("Error: Failed to download data for one or more cryptocurrencies")