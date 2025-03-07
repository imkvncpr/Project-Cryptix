# Models/Bitcoin/__init__.py

# Import base classes/modules from parent package
from .. import CryptoPricePredictor

# Import and expose the Bitcoin predictor class
from .BitC_Predictor import BitC_Predictor

# Define what gets imported with "from Models.Bitcoin import *"
__all__ = ['BitC_Predictor']

# This init file allows you to:
# 1. Import BitC_Predictor directly: from Models.Bitcoin import BitC_Predictor
# 2. Access it through the Models package: from Models import BitC_Predictor (if exposed in Models/__init__.py)
# 3. Run the BitC_Predictor module directly with: python -m Models.Bitcoin.BitC_Predictor