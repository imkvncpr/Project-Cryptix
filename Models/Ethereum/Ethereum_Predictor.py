import sys
import os
# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Now use absolute import
from Models.CryptoPricePredictor import CryptoPricePredictor


class Ethereum_Predictor(CryptoPricePredictor):
    def __init__(self, look_back: int = 60, test_size: float = 0.2):
        super().__init__(ticker='ETH-USD',
                         look_back=look_back,
                         test_size=test_size)
        
if __name__ == '__main__':
    # Example usage
    predictor = Ethereum_Predictor()
    data = predictor.download_data(years=2)
    if data is not None:
        history = predictor.fit(data)
        predictions = predictor.predict(data)
        print(f"Trained model on {len(data)} days of Ethereum data")
        
