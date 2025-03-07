# Models/Tether/__init__.py

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

# Import and expose the Tether predictor class
from .Tether_Predictor import Tether_Prediction

# Define what gets imported with "from Models.Tether import *"
__all__ = ['Tether_Prediction']

# This init file allows you to:
# 1. Import Tether_Prediction directly: from Models.Tether import Tether_Prediction
# 2. Access it through the Models package: from Models import Tether_Prediction (if exposed in Models/__init__.py)
# 3. Run the Tether_Predictor module directly with: python -m Models.Tether.Tether_Predictor