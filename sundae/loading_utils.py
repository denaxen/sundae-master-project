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
    elif model_type == "ar-mt-hf-transformer":
        module_path = "ar_mt_hf_transformer"
        class_name = "ARTransformerHF"
    elif model_type == "mt-hf-sundae":
        # module_path = "mt_hf_sundae"
        module_path = "mt_torch_sundae"
        class_name = "SundaeModel"  
    elif model_type == "toy-mt-hf-sundae":
        module_path = "mt_torch_sundae"
        class_name = "SundaeModel"
    elif model_type == "toy-ar-mt-hf-transformer":
        module_path = "ar_mt_hf_transformer"
        class_name = "ARTransformerHF"
    else:
        raise ValueError(f"Unknown model type {model_type}. Supported types are 'diffusion', 'autoregressive', 'diffusion-mt', 'ar-mt-transformer', 'ar-mt-hf-transformer', and 'mt-hf-transformer'.")
    
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Error loading model class {class_name} from {module_path}: {e}")
        raise
        
    # Instantiate the model
    lightning_module = model_class(config)
    
    return lightning_module 