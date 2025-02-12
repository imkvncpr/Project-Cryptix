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

import matplotlib.pyplot as plt
import seaborn as sns

class TetherPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def get_data(self, start_date='2020-01-01'):
        data = yf.download('USDT-USD', start=start_date)
        return data['Close'].values.reshape(-1, 1)
        
    def prepare_sequences(self, data):
        scaled_data = self.scaler.fit_transform(data)
        X, y = [], []
        
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
        
    def build_model(self):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        self.model = model
        return model
        
    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        return history
        
    def predict_next_day(self, current_sequence):
        scaled_pred = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1))
        return self.scaler.inverse_transform(scaled_pred)[0][0]
        
    def predict_n_days(self, current_sequence, n_days):
        predictions = []
        pred_sequence = current_sequence.copy()
        
        for _ in range(n_days):
            next_pred = self.predict_next_day(pred_sequence)
            predictions.append(next_pred)
            
            pred_sequence = np.roll(pred_sequence, -1)
            pred_sequence[-1] = self.scaler.transform([[next_pred]])[0][0]
            
        return np.array(predictions)
        
    def evaluate_predictions(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae
        }
        
    def plot_predictions(self, actual, predicted, title='USDT Price Predictions'):
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual', color='blue')
        plt.plot(predicted, label='Predicted', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    # Initialize predictor
    predictor = TetherPredictor(sequence_length=60)
    
    # Get and prepare data
    data = predictor.get_data()
    X, y = predictor.prepare_sequences(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Build and train model
    predictor.build_model()
    history = predictor.train(X_train, y_train)
    
    # Make predictions
    y_pred = predictor.model.predict(X_test)
    y_pred = predictor.scaler.inverse_transform(y_pred)
    y_test = predictor.scaler.inverse_transform(y_test)
    
    # Evaluate model
    metrics = predictor.evaluate_predictions(y_test, y_pred)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")
    
    # Future predictions
    last_sequence = X[-1]
    future_predictions = predictor.predict_n_days(last_sequence, 30)
    print("\nNext 30 days predictions:", future_predictions)
    
    # Plot results
    predictor.plot_predictions(y_test, y_pred)

if __name__ == "__main__":
    main()