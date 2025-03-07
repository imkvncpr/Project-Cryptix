import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import yfinance as yf
import logging
import os
from typing import Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CryptoPricePredictor:
    """
    Base class for cryptocurrency price prediction using LSTM neural networks.
    
    This class provides functionality to download historical cryptocurrency data,
    preprocess it, train an LSTM model, and make predictions.
    """
    
    def __init__(self, ticker: str, look_back: int = 60, test_size: float = 0.2):
        """
        Initialize the CryptoPricePredictor.
        
        Args:
            ticker (str): The cryptocurrency ticker symbol (e.g., 'BTC-USD').
            look_back (int): Number of previous days to use for prediction.
            test_size (float): Proportion of data to use for testing.
        """
        self.ticker = ticker
        self.look_back = look_back
        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        
    def download_data(self, years: int = 2) -> Optional[pd.DataFrame]:
        """
        Download historical price data for the specified cryptocurrency.
        
        Args:
            years (int): Number of years of historical data to download.
            
        Returns:
            pd.DataFrame or None: Historical price data or None if download fails.
        """
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=years)
            
            logger.info(f"Downloading {self.ticker} data from {start_date.date()} to {end_date.date()}")
            data = yf.download(self.ticker, start=start_date, end=end_date)
            
            if data.empty:
                logger.error(f"No data downloaded for {self.ticker}")
                return None
                
            logger.info(f"Downloaded {len(data)} days of {self.ticker} data")
            return data
        except Exception as e:
            logger.error(f"Error downloading {self.ticker} data: {str(e)}")
            return None
            
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the data for LSTM model training.
        
        Args:
            data (pd.DataFrame): Historical price data.
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test) preprocessed data arrays.
        """
        # Use closing prices
        df = data[['Close']].copy()
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(df)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test
        
    def build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """
        Build the LSTM model architecture.
        
        Args:
            input_shape (tuple): Shape of input data (time steps, features).
            
        Returns:
            tf.keras.Model: Compiled LSTM model.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(25),
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error'
        )
        
        return model
        
    def fit(self, data: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> tf.keras.callbacks.History:
        """
        Train the LSTM model.
        
        Args:
            data (pd.DataFrame): Historical price data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            
        Returns:
            tf.keras.callbacks.History: Training history.
        """
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        # Build model if it doesn't exist
        if self.model is None:
            input_shape = (X_train.shape[1], 1)
            self.model = self.build_model(input_shape)
            
        # Add early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Add model checkpoint to save best model
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f'best_model_{self.ticker}.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        logger.info(f"Training model for {self.ticker}...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        logger.info(f"Model training completed for {self.ticker}")
        return history
        
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the trained model.
        
        Args:
            data (pd.DataFrame): Historical price data.
            
        Returns:
            np.ndarray: Predicted prices.
        """
        if self.model is None:
            logger.error("Model not trained. Call fit() first.")
            return np.array([])
            
        # Use only closing prices
        df = data[['Close']].copy()
        
        # Scale the data
        scaled_data = self.scaler.transform(df)
        
        # Create sequences for prediction
        X = []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            
        X = np.array(X)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Make predictions
        predicted_scaled = self.model.predict(X)
        
        # Inverse transform to get actual prices
        predicted_prices = self.scaler.inverse_transform(predicted_scaled)
        
        return predicted_prices.flatten()
        
    def predict_future(self, data: pd.DataFrame, days: int = 30) -> np.ndarray:
        """
        Predict future prices for a specified number of days.
        
        Args:
            data (pd.DataFrame): Historical price data.
            days (int): Number of days to predict into the future.
            
        Returns:
            np.ndarray: Predicted future prices.
        """
        if self.model is None:
            logger.error("Model not trained. Call fit() first.")
            return np.array([])
            
        # Get the last 'look_back' days of data
        df = data[['Close']].copy()
        scaled_data = self.scaler.transform(df)
        
        # Use the last look_back days as the initial sequence
        curr_sequence = scaled_data[-self.look_back:].reshape(1, self.look_back, 1)
        
        future_predictions = []
        
        for _ in range(days):
            # Get prediction for next day
            next_pred = self.model.predict(curr_sequence)[0, 0]
            future_predictions.append(next_pred)
            
            # Update sequence by removing the first value and adding the new prediction
            curr_sequence = np.append(
                curr_sequence[:, 1:, :],
                [[next_pred]],
                axis=1
            )
            
        # Convert scaled predictions back to original scale
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_prices = self.scaler.inverse_transform(future_predictions)
        
        return future_prices.flatten()
        
    def evaluate(self, data: pd.DataFrame) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            data (pd.DataFrame): Historical price data.
            
        Returns:
            dict: Evaluation metrics.
        """
        if self.model is None:
            logger.error("Model not trained. Call fit() first.")
            return {}
            
        X_train, X_test, y_train, y_test = self.preprocess_data(data)
        
        # Evaluate the model
        evaluation = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Make predictions on test data
        y_pred = self.model.predict(X_test)
        
        # Inverse transform for actual prices
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = self.scaler.inverse_transform(y_pred)
        
        # Calculate additional metrics
        mse = np.mean((y_pred_inv - y_test_inv) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred_inv - y_test_inv))
        
        return {
            'loss': evaluation,
            'mse': mse[0],
            'rmse': rmse[0],
            'mae': mae[0]
        }
        
    def save_model(self, path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return
            
        try:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            
    def load_model(self, path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            path (str): Path to the saved model.
        """
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")