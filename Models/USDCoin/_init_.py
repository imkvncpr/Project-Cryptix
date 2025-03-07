# Models/USDCoin/__init__.py

# Import base classes/modules from parent package
try:
    from .. import CryptoPricePredictor
except ImportError:
    # Fallback for direct import
    import os
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    sys.path.insert(0, project_root)
    from Models.CryptoPricePredictor import CryptoPricePredictor

# Import and expose the USDCoin predictor class
from .usdC__Predictor import usdC__Predictor

# Define what gets imported with "from Models.USDCoin import *"
__all__ = ['usdC__Predictor']

# This init file allows you to:
# 1. Import usdC__Predictor directly: from Models.USDCoin import usdC__Predictor
# 2. Access it through the Models package: from Models import usdC__Predictor (if exposed in Models/__init__.py)
# 3. Run the USDCoin predictor module directly with: python -m Models.USDCoin.usdC__Predictor