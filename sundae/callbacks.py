import torch
import lightning as L
from loguru import logger

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