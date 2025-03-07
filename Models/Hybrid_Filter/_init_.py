# Models/Hybrid_Filter/Hybrid_Filter_init.py

import os
import sys
import logging

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hybrid_filter_init.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

def initialize_hybrid_filter():
    """
    Initialize and validate Hybrid Filter components
    """
    logger.info("Initializing Hybrid Filter components...")
    
    try:
        # Use absolute imports
        from Models.Hybrid_Filter.Technical_Indicator import Technical_Indicator
        from Models.Hybrid_Filter.Algorithmic_signals import AlgorithmicSignals
        from Models.Hybrid_Filter.Hybrid_Predictor import HybridPredictor
        
        # Instantiate components to verify initialization
        logger.info("Instantiating Technical Indicator...")
        tech_indicator = Technical_Indicator()
        
        logger.info("Instantiating Algorithmic Signals...")
        algo_signals = AlgorithmicSignals()
        
        logger.info("Instantiating Hybrid Predictor...")
        hybrid_predictor = HybridPredictor()
        
        logger.info("All Hybrid Filter components initialized successfully")
        
        return {
            'technical_indicator': tech_indicator,
            'algorithmic_signals': algo_signals,
            'hybrid_predictor': hybrid_predictor
        }
    
    except ImportError as e:
        logger.error(f"Import error during Hybrid Filter initialization: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during Hybrid Filter initialization: {e}")
        raise

def validate_dependencies():
    """
    Check and log key dependencies for Hybrid Filter
    """
    logger.info("Checking Hybrid Filter dependencies...")
    
    dependencies = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('sklearn', 'sklearn'),
        ('tensorflow', 'tf')
    ]
    
    for dep_name, dep_alias in dependencies:
        try:
            module = __import__(dep_name)
            version = getattr(module, '__version__', 'Unknown')
            logger.info(f"{dep_name.capitalize()} version: {version}")
        except ImportError:
            logger.warning(f"{dep_name.capitalize()} is not installed")
        except Exception as e:
            logger.error(f"Error checking {dep_name}: {e}")

def main():
    """
    Main initialization process for Hybrid Filter
    """
    try:
        # Check dependencies
        validate_dependencies()
        
        # Initialize Hybrid Filter components
        components = initialize_hybrid_filter()
        
        return components
    
    except Exception as e:
        logger.error(f"Hybrid Filter initialization failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()