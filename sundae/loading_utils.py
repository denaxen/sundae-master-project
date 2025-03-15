import importlib
from loguru import logger

def get_module(config):
    """Load and instantiate the LightningModule from config."""
    
    # Import the module
    module_path = "models"
    class_name = "SundaeTransformerModule" 
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading model class {class_name} from {module_path}: {e}")
        raise
        
    # Instantiate the model
    lightning_module = model_class(config)
    
    return lightning_module 