import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import lightning as L
from loguru import logger

from x_transformers import TransformerWrapper, Encoder


class SundaeTransformerModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = TransformerWrapper(
            num_tokens=config.data.vocabulary_size,
            max_seq_len=config.data.sequence_length,
            attn_layers=Encoder(
                dim=config.model.embedding_dim,
                depth=config.model.nb_layers,
                head=config.model.nb_heads,
                use_scalenorm=config.model.use_scalenorm,
                ff_glu=config.model.use_glu,
                rotary_pos_emb=config.model.use_rotary,
            )
        )
        
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def get_random_text(self, shape):
        return torch.randint(self.config.data.vocabulary_size, shape)
        
    def corrupt_text(self, batched_text):
        corruption_prob_per_sequence = torch.rand((batched_text.shape[0], 1))
        rand = torch.rand(batched_text.shape)
        mask = (rand < corruption_prob_per_sequence).to(batched_text.device)
        
        random_text = self.get_random_text(batched_text.shape).to(batched_text.device)
        return mask * random_text + ~mask * batched_text
        
    def get_logits(self, batched_text):
        samples = self.corrupt_text(batched_text)
        all_logits = []
        for _ in range(self.config.unroll_steps):
            logits = self.model(samples)
            samples = Categorical(logits=logits).sample().detach()
            all_logits.append(logits)
        final_logits = torch.cat(all_logits, axis=0)
        return final_logits
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        logits = self.get_logits(batch)
        targets = batch.repeat(self.config.unroll_steps, 1)
        accuracy = (logits.argmax(dim=-1) == targets).sum() / targets.numel()
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy * 100.0, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.get_logits(batch)
        targets = batch.repeat(self.config.unroll_steps, 1)
        accuracy = (logits.argmax(dim=-1) == targets).sum() / targets.numel()
        loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy * 100.0, prog_bar=True)
        
        return loss
    
    def sample_text(self, nb_samples=4, min_steps=10):
        """Generate text samples from the model"""
        device = self.device
        
        batched_text = self.get_random_text((nb_samples, self.config.data.sequence_length)).to(device)
        sample_mask = torch.zeros(nb_samples).bool().to(device)
        
        for n in range(self.config.sample.steps):
            old_sample_mask = sample_mask.clone()
            logits = self.model(batched_text[~sample_mask])
            sample = Categorical(logits=logits / self.config.sample.temperature).sample()
            
            mask = (torch.rand(sample.shape) > self.config.sample.sample_proportion).to(device)
            sample[mask] = batched_text[~sample_mask][mask]
            
            if n >= min_steps:
                sample_mask[~sample_mask] = torch.all(
                    (sample == batched_text[~sample_mask]).view(sample.shape[0], -1), dim=-1
                )
                
            if torch.all(sample_mask).item():
                break
                
            batched_text[~old_sample_mask] = sample
            
        logger.info(f"Stopped sampling after {n+1} steps.")
        return batched_text
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        return optimizer 