# Absolute path import method
import os
import sys

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Now use absolute imports
import Models.Hybrid_Filter
import Models.Bitcoin
import Models.Ethereum
import Models.Tether
import Models.USDCoin

from Models.CryptoPricePredictor import CryptoPricePredictor

# Optional: You can add any initialization logic here
def initialize_models():
    print("Initializing CryptixProj Models...")
    # Add any global initialization logic if needed

if __name__ == '__main__':
    initialize_models()