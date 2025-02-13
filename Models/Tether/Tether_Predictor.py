import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Changed from relative import to absolute import
from CryptoPricePredictor import CryptoPricePredictor

class Tether_Prediction(CryptoPricePredictor):
    def __init__(self, look_back: int = 60, test_size: float = 0.2):
        super().__init__(ticker='USDT-USD', 
                         look_back=look_back, 
                         test_size=test_size)
        
if __name__ == '__main__':
    # Example usage
    predictor = Tether_Prediction()
    data = predictor.download_data(years=2)
    if data is not None:
        history = predictor.fit(data)
        predictions = predictor.predict(data)
        print(f"Trained model on {len(data)} days of Tether data")
        