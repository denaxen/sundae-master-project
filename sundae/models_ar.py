import torch
import torch.nn.functional as F
import lightning as L
from loguru import logger
from x_transformers import TransformerWrapper, Decoder

class AutoregressiveTransformerModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create an autoregressive transformer (ensure it applies causal masking)
        # The x_transformers library automatically sets up causal masking when using a decoder or by explicit configuration.
        self.model = TransformerWrapper(
            num_tokens=config.data.vocabulary_size,
            max_seq_len=config.data.sequence_length,
            attn_layers=Decoder(
                dim=config.model.embedding_dim,
                depth=config.model.nb_layers,
                head=config.model.nb_heads,
                use_scalenorm=config.model.use_scalenorm,
                ff_glu=config.model.use_glu,
                rotary_pos_emb=config.model.use_rotary,
            )
        )
        
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def forward(self, x):
        # x is expected to be a batch of token indices with shape (B, seq_len)
        # The transformer returns logits of shape (B, seq_len, vocab_size)
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Standard autoregressive language modeling:
        # for a sequence of tokens [t0, t1, t2, ..., tN-1], predict [t1, t2, ..., tN]
        logits = self.model(batch)
        
        # Remove the last prediction (there is no target for the final token)
        logits = logits[:, :-1, :]
        targets = batch[:, 1:]
        loss = F.cross_entropy(logits.transpose(1, 2), targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean() * 100.0
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits = self.model(batch)
        logits = logits[:, :-1, :]
        targets = batch[:, 1:]
        
        loss = F.cross_entropy(logits.transpose(1, 2), targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean() * 100.0
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)
        return loss
    
    def sample_text(self, nb_samples=4, prompt=None):
        """
        Autoregressively generates text.
        If a prompt is provided, it will continue from the prompt; otherwise, generation starts from random tokens.
        """
        device = self.device
        
        if prompt is None:
            # Start with a random token for each sample
            prompt = torch.randint(0, self.config.data.vocabulary_size, (nb_samples, 1)).to(device)
        else:
            prompt = prompt.to(device)
        
        generated = prompt
        max_new_tokens = self.config.data.sequence_length - generated.shape[1]
        
        for _ in range(max_new_tokens):
            # Get logits for the current sequence and only compute the logits for the last token
            logits = self.model(generated)
            next_logits = logits[:, -1, :]  # (batch_size, vocab_size)
            # Sample from the distribution (or use greedy selection)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            
        # generated: (nb_samples, sequence_length)
        return generated
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.optimizer.learning_rate) 