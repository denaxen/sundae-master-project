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
                heads=config.model.nb_heads,
                use_scalenorm=config.model.use_scalenorm,
                ff_glu=config.model.use_glu,
                rotary_pos_emb=config.model.use_rotary,
            ),
            return_only_embed=True
        )
        
        # Decoder: generate target sentence (non-autoregressive, no causal mask)
        self.decoder = TransformerWrapper(
            num_tokens=config.data.vocabulary_size,
            max_seq_len=config.data.target_sequence_length,
            # the only difference from the encoder is in the causal masking, so we use encder here - don't look at name confusion
            attn_layers=Encoder(
                dim=config.model.embedding_dim,
                depth=config.model.nb_layers,
                heads=config.model.nb_heads,
                use_scalenorm=config.model.use_scalenorm,
                ff_glu=config.model.use_glu,
                rotary_pos_emb=config.model.use_rotary,
                cross_attend=True
            )
        )

        # -------------------------
        # Length Embedding
        # (One embedding vector per length class)
        # -------------------------
        self.length_embed = torch.nn.Embedding(
            num_embeddings=config.model.downsampled_target_length,  # e.g. 64
            embedding_dim=config.model.embedding_dim
        )
        
        # -------------------------
        # 4) Length Predictor
        # (Detaches encoder output to avoid
        #  backprop into the encoder)
        # -------------------------
        self.length_predictor = torch.nn.Sequential(
            torch.nn.Linear(config.model.embedding_dim, config.model.target_length_prediction_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.model.target_length_prediction_hidden_dim, config.model.downsampled_target_length)
        )

        self.decoder.token_emb = self.encoder.token_emb
        
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def corrupt_text(self, batched_text):
        # For each target sequence, sample a corruption probability and corrupt tokens accordingly
        corruption_prob = torch.rand((batched_text.shape[0], 1), device=batched_text.device)
        rand = torch.rand(batched_text.shape, device=batched_text.device)
        mask = (rand < corruption_prob).to(batched_text.device)
        random_text = torch.randint(self.config.data.vocabulary_size, batched_text.shape, device=batched_text.device)
        return mask * random_text + (~mask) * batched_text
    
    def forward(self, src, tgt):
        # Validate input tokens
        max_idx = torch.max(src).item()
        if max_idx > self.config.data.vocabulary_size:
            raise ValueError(f"Found token ID {max_idx} in source text, but vocabulary size is {self.config.data.vocabulary_size}")
        
        # Encode source sentence
        src_enc = self.encoder(src)

        # 2) DETACH for length predictor
        # so length loss won't backprop to encoder
        with torch.no_grad():
            src_enc_detached = src_enc.clone().detach()
        
         # 3) Predict length from the (mean-pooled) detached encoding
        pred_length_logits = self.length_predictor(src_enc_detached.mean(dim=1))

        # 4) Compute ground-truth length classes (downsampled)
        #    Then create a length embedding (teacher forcing)
        gt_len = (tgt != self.config.data.pad_token).sum(dim=1)  # actual target length
        gt_len_downsampled = torch.clamp((gt_len + 1) // 2, max=self.config.model.downsampled_target_length - 1)
        length_emb = self.length_embed(gt_len_downsampled)       # shape: [batch_size, d_model]

        # 5) Prepend length embedding to encoder output
        length_emb = length_emb.unsqueeze(1)                     # shape: [batch_size, 1, d_model]
        src_enc_with_len = torch.cat([length_emb, src_enc], dim=1)
        
        # Begin with a corrupted version of the target text
        current_tgt = self.corrupt_text(tgt)
        all_logits = []
        
        # Unrolled denoising steps
        for _ in range(self.config.unroll_steps):
            # Decode using the current target and conditioning on the source encoding
            logits = self.decoder(current_tgt, context=src_enc_with_len)
            # Sample new tokens from the predicted distribution (detach to prevent gradient flow)
            current_tgt = Categorical(logits=logits).sample().detach()
            all_logits.append(logits)
        
        # Concatenate logits from all unroll steps for loss computation
        final_logits = torch.cat(all_logits, dim=0)
        return final_logits, pred_length_logits
    
    def training_step(self, batch, batch_idx):
        # Assume batch is a tuple: (src, tgt)
        src, tgt = batch['source'], batch['target']
        logits, pred_length_logits = self.forward(src, tgt)
        
        # Repeat ground truth target for each unroll step along the batch dimension
        repeated_tgt = tgt.repeat(self.config.unroll_steps, 1)
        token_loss = F.cross_entropy(logits.permute(0, 2, 1), repeated_tgt, label_smoothing=self.config.model.label_smoothing)
        
        # Compute ground truth target lengths (downsampled if necessary)
        # Here, we assume pad tokens are given by config.data.pad_token
        gt_len = (tgt != self.config.data.pad_token).sum(dim=1)
        gt_len_downsampled = torch.clamp((gt_len + 1) // 2, max=self.config.model.downsampled_target_length - 1)
        length_loss = F.cross_entropy(pred_length_logits, gt_len_downsampled)
        
        
        loss = token_loss + self.config.model.length_loss_weight * length_loss
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_token_loss', token_loss, prog_bar=True)
        self.log('train_length_loss', length_loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch['source'], batch['target']
        logits, pred_length_logits = self.forward(src, tgt)
        repeated_tgt = tgt.repeat(self.config.unroll_steps, 1)
        token_loss = F.cross_entropy(logits.permute(0, 2, 1), repeated_tgt, label_smoothing=self.config.model.label_smoothing)
        gt_len = (tgt != self.config.data.pad_token).sum(dim=1)
        gt_len_downsampled = torch.clamp((gt_len + 1) // 2, max=self.config.model.downsampled_target_length - 1)
        length_loss = F.cross_entropy(pred_length_logits, gt_len_downsampled)
        loss = token_loss + self.config.model.length_loss_weight * length_loss
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def sample_translation(self, src, min_steps=4):
        """Generate translation for a given source sentence batch."""
        src_enc = self.encoder(src)

        # Predict length from src_enc (no teacher forcing)
        pred_length_logits = self.length_predictor(src_enc.mean(dim=1))
        pred_len_class = pred_length_logits.argmax(dim=-1)  # shape: [batch_size]

        # Turn it into embedding
        length_emb = self.length_embed(pred_len_class)       # shape: [batch_size, d_model]
        length_emb = length_emb.unsqueeze(1)                # shape: [batch_size, 1, d_model]
        src_enc_with_len = torch.cat([length_emb, src_enc], dim=1)
        
        # Initialize random target of "predicted" length
        batch_size = src.shape[0]
        max_len = self.config.data.target_sequence_length
        init_tgt = torch.randint(
            self.config.data.vocabulary_size,
            (batch_size, max_len),
            device=src.device
        )
        # Optionally set tokens beyond predicted length to pad
        predicted_len_upsampled = pred_len_class * 2
        for i in range(batch_size):
            length_i = predicted_len_upsampled[i]
            if length_i < max_len:
                init_tgt[i, length_i:] = self.config.data.pad_token

        # Iterative refinement
        for step_idx in range(self.config.sample.steps):
            logits = self.decoder(init_tgt, context=src_enc_with_len)
            sample = Categorical(logits=logits / self.config.sample.temperature).sample()
            init_tgt = sample
        
        logger.info(f"Stopped sampling after {step_idx+1} steps.")
        return init_tgt
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
            lr=self.config.optimizer.learning_rate, 
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay)
        return optimizer
