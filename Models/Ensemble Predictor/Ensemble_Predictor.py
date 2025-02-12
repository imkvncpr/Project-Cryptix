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
from Bitcoin import BitC_Predictor
from Ethereum import Ethereum_Predictor
from Models.Tether import Tether_Predictior
from USDCoin import usdC__Predictor

class EnsemblePredictor:
    def __init__(self, look_back: int = 60, test_size: float = 0.2):
        self.look_back = look_back
        self.test_size = test_size
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.models = {
            'BTC': BitC_Predictor(look_back=look_back, test_size=test_size),
            'ETH': Ethereum_Predictor(look_back=look_back, test_size=test_size),
            'USDT': Tether_Predictior(look_back=look_back, test_size=test_size),
            'USDC': usdC__Predictor(look_back=look_back, test_size=test_size)
        }
        self.model = None
        self.look_back = None
        
def get_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[np.array, np.array]:
        X, y = [], []
        for coin, df in data.items():
            X_, y_ = self.models[coin].get_data(df)
            X.append(X_)
            y.append(y_)
        X = np.concatenate(X, axis=1)
        y = np.concatenate(y, axis=1)
        return X, y
    
def create_model(self) -> tf.keras.models.Model:
        inputs, outputs = [], []
        for coin, model in self.models.items():
            inputs.append(model.model.input)
            outputs.append(model.model.output)

        merged = Concatenate()(outputs)
        merged = Dense(50,activation='relu')(merged)
        merged = Dense(25, activation='relu')(merged)
        merged = Dense(1)(merged)
        
        model = Model(inputs=inputs, outputs=merged)
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
def fit(self, data: Dict[str, pd.DataFrame], epochs: int = 100, batch_size: int = 32):
        X, y = self.get_data(data)
        self.model = self.create_model()
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        
        for coin, model in self.models.items():
            model.fit(data[coin], epochs=epochs, batch_size=batch_size)
            
def predict(self, data: Dict[str, pd.DataFrame], days: int = 30) -> Dict[str, np.array]:
    """
    Make predictions using train Models
    """
    if self.model is None:
        raise Exception("Model not trained yet")
    
    x, = self.get_data(data)
    predictions = self.model.predict(x)
    return self.scaler.inverse_transform(predictions(-1, 1))

def evaluate(self, data: Dict[str, pd.DataFrame]):
    """
    Evaluate the model using test data
    """
    if self.model is None:
        raise Exception("Model not trained yet")
    
    x, y = self.get_data(data)
    loss = self.model.evaluate(x, y)
    return loss

def predict_future(self, data: Dict[str, pd.DataFrame], days: int = 30) -> Dict[str, np.array]:
    """
    Predict future prices
    """
    predictions = {}
    last_sequence = {}
    
    # get last sequence for each coin
    
    for coin, model in self.models.items():
        last_data = data[coin].tail(model.look_back)
        last_sequence[coin] = model.get_last_sequence(data[coin])
        
   #  make predictions for the next days
   
    ensemble_input = np.concatenate(list(last_sequence()), axis=1)
    ensmble_pred = self.model.predict(ensemble_input)
    
#    make predictions for each coin
    for coin,model in self.models.items():
        pred = model.predict(data[coin].tail(model.look_back))
        predictions[coin] = {
            'individual': float(pred),
            'ensemble': float(ensmble_pred[0])
        }
        return predictions
    
def plot_predictions(self, actual_data: Dict[str, pd.DataFrame], predictions: Dict[str, np.array]):
    """
    Plot the actual and predicted prices
    """
    fig = go.Figure()
    
    for coin in self.models.keys():
        # plot actual data
        fig.add_trace(go.Scatter(
            x = actual_data[coin].index,
            y = actual_data[coin]['Close'],
            mode='lines',
            name=f'{coin} Actual'
        ))
        
        # plot predictions
        if coin in predictions:
            fig.add_trace(go.Scatter(
                x = actual_data[coin].index,
                y = predictions[coin],
                mode='lines',
                name=f'{coin} Predicted',
                line=dict(dash='dash')
            ))
        fig.add_trace(go.Scatter(
            x = actual_data[coin].index,
            y = predictions[coin],
            mode='lines',
            name=f'{coin} Predicted'
        ))
        
        fig.show()
        
def save_models(self, path: str = 'models/ensemble'):
    """
    Save the individual models
    """
    for coin, model in self.models.items():
        model.save_model(f'{path}_{coin}')
        
    self.model.save(f'{path}_ensemble.h5')
    
    "save individual models"
    for coin, model in self.models.items():
        model.save_model(f'{path}_{coin}')
        
    self.model.save(f'{path}_ensemble.h5')
    
def load_models(self, path: str = 'models/ensemble'):
    """
    Load the individual models
    """
    for coin, model in self.models.items():
        model.load_model(f'{path}_{coin}')
        
    self.model = tf.keras.models.load_model(f'{path}_ensemble.h5')
    
    for coin, model in self.models.items():
        model.load_model(f'{path}_{coin}')
        
    self.model = tf.keras.models.load_model(f'{path}_ensemble.h5')
        
    
    
    
