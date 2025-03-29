import torch
import lightning as L
from loguru import logger
from transformers import AutoTokenizer

class TextSamplingCallback(L.Callback):
    def __init__(self, sample_frequency=500, nb_samples=4):
        super().__init__()
        self.sample_frequency = sample_frequency  # Now represents steps, not epochs
        self.nb_samples = nb_samples
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Generate text samples every N steps."""
        # Check if we've reached a multiple of sample_frequency steps
        if trainer.global_step > 0 and trainer.global_step % self.sample_frequency == 0:
            logger.info(f"Sampling from current model at step {trainer.global_step}")
            
            # Make sure we're in eval mode for sampling
            was_training = pl_module.training
            if was_training:
                pl_module.eval()
            
            with torch.no_grad():
                samples = pl_module.sample_text(nb_samples=self.nb_samples)
            
            # Return to training mode if we were in it
            if was_training:
                pl_module.train()
            
            # Get dataset from the validation DataLoader
            val_loader = trainer.val_dataloaders
            dataset = val_loader.dataset
            
            # Display samples
            for i, sample in enumerate(samples):
                logger.info(f"- " + ''.join(dataset.id_token[i.item()] for i in sample)) 

class TranslationSamplingCallback(L.Callback):
    def __init__(self, sample_frequency=500, nb_samples=4):
        super().__init__()
        self.sample_frequency = sample_frequency
        self.nb_samples = nb_samples
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Generate translation samples every N steps."""
        # Check if we've reached a multiple of sample_frequency steps
        if trainer.global_step > 0 and trainer.global_step % self.sample_frequency == 0:
            logger.info(f"Sampling translations at step {trainer.global_step}")
            
            # Make sure we're in eval mode for sampling
            was_training = pl_module.training
            if was_training:
                pl_module.eval()
            
            # Get dataset from the validation DataLoader - note it's not a list
            val_dataloader = trainer.val_dataloaders
            val_dataset = val_dataloader.dataset
            
            # Get the shared tokenizer
            tokenizer_path = pl_module.config.data.shared_tokenizer_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Sample from validation set
            with torch.no_grad():
                # Get some random samples from validation set
                val_samples = []
                for _ in range(self.nb_samples):
                    idx = torch.randint(0, len(val_dataset), (1,)).item()
                    val_samples.append(val_dataset[idx])
                
                for sample in val_samples:
                    # Put source on device and add batch dimension
                    src = sample['source'].unsqueeze(0).to(pl_module.device)
                    
                    # Generate translation
                    translation = pl_module.sample_translation(src)
                    
                    # Decode using shared tokenizer
                    decoded_text = tokenizer.decode(translation[0].tolist(), skip_special_tokens=True)
                    
                    # Log the results
                    source_text = tokenizer.decode(sample['source'].tolist(), skip_special_tokens=True)
                    reference_text = tokenizer.decode(sample['target'].tolist(), skip_special_tokens=True)
                    
                    logger.info(f"Source: {source_text}")
                    logger.info(f"Generated: {decoded_text}")
                    logger.info(f"Reference: {reference_text}")
                    logger.info(f"Generated tokens: {translation}")
                    # Count non-pad tokens in target
                    pad_token_id = pl_module.config.data.get("pad_token", 1)
                    target_length = (sample['target'] != pad_token_id).sum().item()
                    logger.info(f"Target length (excluding pad tokens): {target_length}")
                    generated_length = (translation[0] != pad_token_id).sum().item()
                    logger.info(f"Generated length (excluding pad tokens): {generated_length}")
                    logger.info("----")
            
            # Return to training mode if needed
            if was_training:
                pl_module.train() 


class ValEveryNStepsCallback(L.Callback):
    def __init__(self, test_interval: int = 100):
        super().__init__()
        self.test_interval = test_interval

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only run test every test_interval steps (and skip step 0)
        if trainer.global_step > 0 and trainer.global_step % self.test_interval == 0:
            # Get the validation dataloader from the trainer
            val_dataloader = trainer.val_dataloaders
            
            # Use test() and explicitly provide the dataloader
            trainer.validate(pl_module, dataloaders=val_dataloader, verbose=True)