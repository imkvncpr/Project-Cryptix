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
import matplotlib.pyplot as plt
import seaborn as sns

class USDCPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def get_data(self, start_date='2020-01-01'):
        data = yf.download('USDC-USD', start=start_date)
        return data['Close'].values.reshape(-1, 1)
    
    def prepare_sequences(self, data):
        scaled_data = self.scaler.fit_transform(data)
        x, y = [], []
        
        for i in range(len(scaled_data) - self.sequence_length):
            x.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length])
            
        return np.array(x), np.array(y)
    
    def build_model(self):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        self.model = model
        return model
    
    def train(self, x, y, epochs=100, batch_size=32, validation_split=0.2):
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = self.model.fit(
            x, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history

    def predict_next_day(self, current_sequence):
        scaled_prediction = self.model.predict(current_sequence.reshape(1, self.sequence_length, 1))
        return self.scaler.inverse_transform(scaled_prediction)[0][0]
    
    def predict_next_days(self, current_sequence, n=7):
        predictions = []
        sequence = current_sequence.copy()
        
        for _ in range(n):
            scaled_sequence = self.scaler.transform(sequence.reshape(-1, 1))[-self.sequence_length:]
            prediction = self.predict_next_day(scaled_sequence)
            predictions.append(prediction)
            sequence = np.append(sequence[1:], prediction)
            
        return np.array(predictions)
    
    def make_predictions(self, x_test):
        """Make predictions for all test sequences"""
        predictions = []
        for sequence in x_test:
            # Predict next value for each sequence
            scaled_pred = self.model.predict(sequence.reshape(1, self.sequence_length, 1))
            pred = self.scaler.inverse_transform(scaled_pred)[0][0]
            predictions.append(pred)
        return np.array(predictions)
    
    def evaluate_predictions(self, actual, predicted):
        return np.mean(np.abs(actual - predicted))
    
    def plot_results(self, actual, predicted, title="USDC Price Prediction"):
        plt.figure(figsize=(14, 7))
        plt.plot(self.scaler.inverse_transform(actual), label='Actual', color='blue')
        plt.plot(predicted, label='Predicted', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def main(self):
        # Get and prepare data
        data = self.get_data()
        x, y = self.prepare_sequences(data)
        
        # Split data
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
        
        # Train model
        self.build_model()
        history = self.train(x_train, y_train)
        
        # Make predictions on test set
        test_predictions = self.make_predictions(x_test)
        
        # Plot results and calculate error for test set predictions
        self.plot_results(y_test, test_predictions, "Test Set Predictions")
        test_error = self.evaluate_predictions(y_test, self.scaler.transform(test_predictions.reshape(-1, 1)))
        print(f'Test Set Mean Absolute Error: {test_error}')
        
        # Make future predictions
        last_sequence = x_test[-1]
        future_predictions = self.predict_next_days(self.scaler.inverse_transform(last_sequence))
        print("\nPredicted prices for next 7 days:")
        for i, pred in enumerate(future_predictions, 1):
            print(f"Day {i}: ${pred:.4f}")

if __name__ == '__main__':
    predictor = USDCPredictor()
    predictor.main()