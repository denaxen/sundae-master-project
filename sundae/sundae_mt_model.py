import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import lightning as L
from x_transformers import TransformerWrapper, Encoder, Decoder
from loguru import logger

class SundaeMTModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Encoder: process the source sentence
        self.encoder = TransformerWrapper(
            num_tokens=config.data.vocabulary_size,
            max_seq_len=config.data.source_sequence_length,
            attn_layers=Encoder(
                dim=config.model.embedding_dim,
                depth=config.model.nb_layers,
                head=config.model.nb_heads,
                use_scalenorm=config.model.use_scalenorm,
                ff_glu=config.model.use_glu,
                rotary_pos_emb=config.model.use_rotary,
            )
        )
        
        # Decoder: generate target sentence (non-autoregressive, no causal mask)
        self.decoder = TransformerWrapper(
            num_tokens=config.data.vocabulary_size,
            max_seq_len=config.data.target_sequence_length,
            attn_layers=Decoder(
                dim=config.model.embedding_dim,
                depth=config.model.nb_layers,
                head=config.model.nb_heads,
                use_scalenorm=config.model.use_scalenorm,
                ff_glu=config.model.use_glu,
                rotary_pos_emb=config.model.use_rotary,
            )
        )
        
        # Target length predictor: predicts (downsampled) target length from source encoding
        self.length_predictor = torch.nn.Sequential(
            torch.nn.Linear(config.model.embedding_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, config.data.downsampled_target_length)  # number of classes for length
        )
        
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def corrupt_text(self, batched_text):
        # For each target sequence, sample a corruption probability and corrupt tokens accordingly
        corruption_prob = torch.rand((batched_text.shape[0], 1), device=batched_text.device)
        rand = torch.rand(batched_text.shape, device=batched_text.device)
        mask = (rand < corruption_prob).to(batched_text.device)
        random_text = torch.randint(self.config.data.vocabulary_size, batched_text.shape, device=batched_text.device)
        return mask * random_text + (~mask) * batched_text
    
    def forward(self, src, tgt):
        # Encode source sentence
        src_enc = self.encoder(src)
        
        # Predict target length (for teacher forcing during training)
        # Here we use a simple pooling (mean) over encoder outputs
        pred_length_logits = self.length_predictor(src_enc.mean(dim=1))
        
        # Begin with a corrupted version of the target text
        current_tgt = self.corrupt_text(tgt)
        all_logits = []
        
        # Unrolled denoising steps
        for _ in range(self.config.unroll_steps):
            # Decode using the current target and conditioning on the source encoding
            logits = self.decoder(current_tgt, context=src_enc)
            # Sample new tokens from the predicted distribution (detach to prevent gradient flow)
            current_tgt = Categorical(logits=logits).sample().detach()
            all_logits.append(logits)
        
        # Concatenate logits from all unroll steps for loss computation
        final_logits = torch.cat(all_logits, dim=0)
        return final_logits, pred_length_logits
    
    def training_step(self, batch, batch_idx):
        # Assume batch is a tuple: (src, tgt)
        src, tgt = batch
        logits, pred_length_logits = self.forward(src, tgt)
        
        # Repeat ground truth target for each unroll step along the batch dimension
        repeated_tgt = tgt.repeat(self.config.unroll_steps, 1)
        token_loss = F.cross_entropy(logits.permute(0, 2, 1), repeated_tgt, label_smoothing=self.config.training.label_smoothing)
        
        # Compute ground truth target lengths (downsampled if necessary)
        # Here, we assume pad tokens are given by config.data.pad_token
        gt_lengths = (tgt != self.config.data.pad_token).sum(dim=1) // 2  # adjust if downsampling factor is 2
        length_loss = F.cross_entropy(pred_length_logits, gt_lengths)
        
        loss = token_loss + self.config.training.length_loss_weight * length_loss
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_token_loss', token_loss, prog_bar=True)
        self.log('train_length_loss', length_loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        logits, pred_length_logits = self.forward(src, tgt)
        repeated_tgt = tgt.repeat(self.config.unroll_steps, 1)
        token_loss = F.cross_entropy(logits.permute(0, 2, 1), repeated_tgt, label_smoothing=self.config.training.label_smoothing)
        gt_lengths = (tgt != self.config.data.pad_token).sum(dim=1) // 2
        length_loss = F.cross_entropy(pred_length_logits, gt_lengths)
        loss = token_loss + self.config.training.length_loss_weight * length_loss
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def sample_translation(self, src, min_steps=4):
        """Generate translation for a given source sentence batch."""
        src_enc = self.encoder(src)
        pred_length_logits = self.length_predictor(src_enc.mean(dim=1))
        # Predict target length (upsample if needed)
        predicted_length = (pred_length_logits.argmax(dim=-1) * 2)  # adjust factor according to training
        
        batch_size = src.shape[0]
        # Initialize target sequences with random tokens, then mask positions beyond predicted length as pad tokens
        init_tgt = torch.randint(self.config.data.vocabulary_size, 
                                 (batch_size, self.config.data.target_sequence_length), 
                                 device=src.device)
        for i in range(batch_size):
            init_tgt[i, predicted_length[i]:] = self.config.data.pad_token
        
        # Iteratively refine target translations
        for n in range(self.config.sample.steps):
            logits = self.decoder(init_tgt, context=src_enc)
            sample = Categorical(logits=logits / self.config.sample.temperature).sample()
            # Here you might choose to update only a subset of tokens based on uncertainty
            init_tgt = sample
        logger.info(f"Stopped sampling after {n+1} steps.")
        return init_tgt
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.training.learning_rate, 
                                     betas=(0.9, 0.999), eps=1e-6, weight_decay=self.config.training.weight_decay)
        return optimizer
