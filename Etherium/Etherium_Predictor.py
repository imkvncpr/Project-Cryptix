import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import plotly.graph_objects as go
import logging
from typing import Tuple, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EthereumPricePredictor:
    def __init__(self, look_back: int = 60, test_size: float = 0.2):
        self.ticker = 'ETH-USD'
        self.look_back = look_back
        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.last_sequence = None

    def download_eth_data(self, years: int = 5) -> Optional[pd.DataFrame]:
        try:
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.DateOffset(years=years)
            data = yf.download(self.ticker, start=start_date, end=end_date)
            return data
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            return None

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data_processed = data[feature_columns].values
        scaled_data = self.scaler.fit_transform(data_processed)
        
        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i])
            y.append(scaled_data[i, 3])
        
        X, y = np.array(X), np.array(y)
        self.last_sequence = X[-1:]
        return X, y

    def create_model(self) -> Sequential:
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.look_back, 5)),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.15),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Sequential, Dict[str, float]]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        
        model = self.create_model()
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        predictions = model.predict(X_test, verbose=0)
        metrics = self._calculate_metrics(y_test, predictions)
        
        self.model = model
        return model, metrics

    def predict_future(self, days: int = 30) -> np.ndarray:
        if self.model is None or self.last_sequence is None:
            raise ValueError("Model must be trained before making predictions")
            
        current_sequence = np.copy(self.last_sequence)
        future_predictions = []
        
        for _ in range(days):
            next_pred = self.model.predict(current_sequence, verbose=0)[0]
            future_predictions.append(next_pred)
            
            new_sequence = np.copy(current_sequence)
            new_sequence[0, :-1] = new_sequence[0, 1:]
            new_sequence[0, -1] = new_sequence[0, -2]
            new_sequence[0, -1, 3] = next_pred
            
            current_sequence = new_sequence
        
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions_reshaped = np.hstack([
            np.zeros((len(future_predictions), 3)),
            future_predictions,
            np.zeros((len(future_predictions), 1))
        ])
        
        return self.scaler.inverse_transform(future_predictions_reshaped)[:, 3]

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true_orig = self.scaler.inverse_transform(np.column_stack([
            np.zeros((len(y_true), 3)),
            y_true.reshape(-1, 1),
            np.zeros((len(y_true), 1))
        ]))[:, 3]
        
        y_pred_orig = self.scaler.inverse_transform(np.column_stack([
            np.zeros((len(y_pred), 3)),
            y_pred.reshape(-1, 1),
            np.zeros((len(y_pred), 1))
        ]))[:, 3]
        
        return {
            'MSE': mean_squared_error(y_true_orig, y_pred_orig),
            'RMSE': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
            'MAE': mean_absolute_error(y_true_orig, y_pred_orig),
            'R2': r2_score(y_true_orig, y_pred_orig)
        }

    def visualize_predictions(self, historical_data: pd.DataFrame, future_prices: np.ndarray):
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            mode='lines',
            name='Historical Prices'
        ))
        
        # Future predictions
        future_dates = pd.date_range(
            start=historical_data.index[-1],
            periods=len(future_prices) + 1,
            closed='right'
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_prices,
            mode='lines',
            name='Predicted Prices',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='Ethereum Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark'
        )
        
        return fig

def main():
    st.set_page_config(page_title="Ethereum Price Predictor", layout="wide")
    st.title("Ethereum Price Predictor")
    
    predictor = EthereumPricePredictor()
    
    # Session state initialization
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # Sidebar controls
    st.sidebar.header("Prediction Controls")
    
    # Data download section
    if st.sidebar.button("Download ETH Data"):
        with st.spinner("Downloading Ethereum historical data..."):
            st.session_state.data = predictor.download_eth_data()
            if st.session_state.data is not None:
                st.success("Data downloaded successfully!")
                st.subheader("Historical Data Preview")
                st.dataframe(st.session_state.data.tail())
    
    # Model training section
    if st.session_state.data is not None and st.sidebar.button("Train Model"):
        with st.spinner("Training LSTM model..."):
            try:
                X, y = predictor.prepare_data(st.session_state.data)
                model, metrics = predictor.train_model(X, y)
                st.session_state.model_trained = True
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("MSE", f"{metrics['MSE']:.2f}")
                col2.metric("RMSE", f"{metrics['RMSE']:.2f}")
                col3.metric("MAE", f"{metrics['MAE']:.2f}")
                col4.metric("R2 Score", f"{metrics['R2']:.4f}")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
    
    # Prediction section
    if st.session_state.model_trained:
        st.subheader("Generate Price Predictions")
        days = st.slider("Number of days to predict", 1, 90, 30)
        
        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                try:
                    future_prices = predictor.predict_future(days=days)
                    
                    # Display predictions table
                    pred_df = pd.DataFrame({
                        'Day': range(1, len(future_prices) + 1),
                        'Predicted Price (USD)': future_prices
                    })
                    st.dataframe(pred_df)
                    
                    # Display interactive plot
                    fig = predictor.visualize_predictions(st.session_state.data, future_prices)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating predictions: {str(e)}")
    
    # Instructions if no data is loaded
    if st.session_state.data is None:
        st.info("👈 Start by downloading Ethereum data using the sidebar button")

if __name__ == "__main__":
    main()