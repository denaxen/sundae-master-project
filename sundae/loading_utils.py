import importlib
from loguru import logger

def get_module(config):
    """Load and instantiate the LightningModule from config based on model type."""
    model_type = getattr(config.model, "type", "diffusion").lower()

    if model_type == "ar-like-sundae-generation":
        module_path = "models_ar"
        class_name = "AutoregressiveTransformerModule"
    elif model_type == "sundae-diffusion-generation":
        module_path = "models"
        class_name = "SundaeTransformerModule"
    elif model_type == "sundae-diffusion-mt":
        module_path = "models_mt_sundae"
        class_name = "SundaeMTModule"
    elif model_type == "ar-mt-transformer":
        module_path = "ar_mt_transformer"
        class_name = "ARTransformerBase"
    else:
        raise ValueError(f"Unknown model type {model_type}. Supported types are 'diffusion' and 'autoregressive'.")
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading model class {class_name} from {module_path}: {e}")
        raise
        
    # Instantiate the model
    lightning_module = model_class(config)
    
    return lightning_module 