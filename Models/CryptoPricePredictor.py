import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
Model = tf.keras.models.Model
Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
LSTM = tf.keras.layers.LSTM
Concatenate = tf.keras.layers.Concatenate
Input = tf.keras.layers.Input
Dropout = tf.keras.layers.Dropout
EarlyStopping = tf.keras.callbacks.EarlyStopping
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
Adam = tf.keras.optimizers.Adam
from typing import Tuple, Optional, Dict
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoPricePredictor:
    def __init__(self, 
                 ticker: str,
                 look_back: int = 60,
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the cryptocurrency price predictor.
        
        Args:
            ticker: Trading symbol for the cryptocurrency (e.g., 'BTC-USD')
            look_back: Number of previous time steps to use for prediction
            test_size: Proportion of dataset to use for testing
            random_state: Random seed for reproducibility
        """
        self.ticker = ticker
        self.look_back = look_back
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.X = None
        self.y = None
        self.last_sequence = None

    def download_data(self, years: int = 5) -> Optional[pd.DataFrame]:
        """
        Download historical cryptocurrency data.
        """
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=years)
            data = yf.download(self.ticker, start=start_date, end=end_date)
            logger.info(f"Successfully downloaded {years} years of {self.ticker} data")
            return data
        except Exception as e:
            logger.error(f"Error downloading {self.ticker} data: {str(e)}")
            return None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model training.
        """
        try:
            # Use features: Open, High, Low, Close, Volume
            feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data_processed = data[feature_columns].values
            
            # Scale features
            scaled_data = self.scaler.fit_transform(data_processed)
            
            X, y = [], []
            for i in range(self.look_back, len(scaled_data)):
                X.append(scaled_data[i-self.look_back:i])
                y.append(scaled_data[i, 3])  # Index 3 is Close price
            
            X, y = np.array(X), np.array(y)
            self.last_sequence = X[-1:]  # Store last sequence for future predictions
            
            logger.info(f"Prepared data shapes - X: {X.shape}, y: {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

    def create_model(self) -> tf.keras.models.Sequential:
        """
        Create LSTM model architecture.
        """
        try:
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(self.look_back, 5)),
                Dropout(0.3),
                LSTM(50),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            logger.info(f"Created model for {self.ticker}")
            return model
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise

    def fit(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the model on provided data.
        """
        try:
            X, y = self.prepare_data(data)
            self.X, self.y = X, y
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            self.model = self.create_model()
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(f'best_model_{self.ticker}.h5', monitor='val_loss', save_best_only=True)
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            logger.info(f"Completed training for {self.ticker}")
            return history
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        try:
            X, _ = self.prepare_data(data)
            predictions = self.model.predict(X)
            
            # Reshape predictions for inverse transform
            pred_reshaped = np.zeros((len(predictions), 5))
            pred_reshaped[:, 3] = predictions.flatten()  # Put predictions in Close price column
            
            # Inverse transform to get actual prices
            predictions_actual = self.scaler.inverse_transform(pred_reshaped)[:, 3]
            
            return predictions_actual
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def predict_future(self, data: pd.DataFrame, days: int = 30) -> np.ndarray:
        """
        Predict future prices for specified number of days.
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
            
        try:
            # Get the last sequence
            last_sequence = self.get_last_sequence(data)
            predictions = []
            
            current_sequence = last_sequence.copy()
            for _ in range(days):
                # Predict next day
                pred = self.model.predict(current_sequence)[0, 0]
                predictions.append(pred)
                
                # Update sequence for next prediction
                new_sequence = current_sequence.copy()
                new_sequence[0, :-1] = new_sequence[0, 1:]
                new_sequence[0, -1, 3] = pred  # Update Close price
                current_sequence = new_sequence
            
            # Convert predictions to actual prices
            pred_reshaped = np.zeros((len(predictions), 5))
            pred_reshaped[:, 3] = predictions
            predictions_actual = self.scaler.inverse_transform(pred_reshaped)[:, 3]
            
            return predictions_actual
        except Exception as e:
            logger.error(f"Error making future predictions: {str(e)}")
            raise

    def get_last_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """
        Get the last sequence from the data for predictions.
        """
        X, _ = self.prepare_data(data)
        return X[-1:]

    def evaluate(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the model performance.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        try:
            X, y = self.prepare_data(data)
            loss = self.model.evaluate(X, y, verbose=0)
            return {'loss': loss}
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def save_model(self, path: str):
        """
        Save the trained model.
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            self.model.save(path)
            logger.info(f"Saved model to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str):
        """
        Load a trained model.
        """
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Loaded model from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise