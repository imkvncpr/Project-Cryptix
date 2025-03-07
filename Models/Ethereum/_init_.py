# Models/Ethereum/__init__.py

# Import base classes/modules from parent package
from .. import CryptoPricePredictor

# Import and expose the Ethereum predictor class
from .Ethereum_Predictor import Ethereum_Predictor

# Define what gets imported with "from Models.Ethereum import *"
__all__ = ['Ethereum_Predictor']

# This init file allows you to:
# 1. Import Ethereum_Predictor directly: from Models.Ethereum import Ethereum_Predictor
# 2. Access it through the Models package: from Models import Ethereum_Predictor (if exposed in Models/__init__.py)
# 3. Run the Ethereum_Predictor module directly with: python -m Models.Ethereum.Ethereum_Predictor