import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import AutoTokenizer
import math
from loguru import logger
import sacrebleu
import nltk
from torch.distributions.categorical import Categorical

class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1   = nn.Linear(dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, dim)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        # simple two‐layer bottleneck with skip
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        return self.act(x + h)

class LengthPredictor(nn.Module):
    """
    6‐block residual predictor of downsampled target length.
    Input: pooled encoder embedding of size E
    Output: logits over Dd length classes
    """
    def __init__(self, embed_dim, hidden_dim, num_blocks, num_classes):
        super().__init__()
        # optional "stem" to mix features (could be identity)
        self.stem   = nn.Linear(embed_dim, embed_dim)
        # six residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(embed_dim, hidden_dim)
            for _ in range(num_blocks)
        ])
        # final classifier to length‐ids
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: (B, E)
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.classifier(h)  # (B, num_classes)

class SundaeModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.pad_token = config.data.pad_token
        # Load the shared tokenizer using the provided path.
        self.tokenizer = AutoTokenizer.from_pretrained(config.data.shared_tokenizer_path)
        
        # For validation outputs collection
        self.my_val_outputs = []
        
        # Determine max position embeddings from source and target lengths.
        max_position = max(config.data.source_sequence_length, config.data.target_sequence_length)
        
        V = config.data.vocabulary_size
        E = config.model.embedding_dim
        N = max(config.data.source_sequence_length, config.data.target_sequence_length)

        # token + position embeddings (shared)
        self.token_emb   = nn.Embedding(V, E, padding_idx=config.data.pad_token)
        self.pos_emb     = nn.Embedding(N, E)
        self.dropout     = nn.Dropout(config.model.dropout)

        # ——— length prediction head ———
        H = config.model.target_length_prediction_hidden_dim
        Dd = config.model.downsampled_target_length  # e.g. 64

        # embedding for the (downsampled) length to prepend
        self.length_pred = LengthPredictor(
            embed_dim=E,
            hidden_dim=H,
            num_blocks=6,     # six residual blocks
            num_classes=Dd
        )
        self.length_emb  = nn.Embedding(Dd, E)
        self.length_loss_weight = config.model.length_loss_weight

        self.model = nn.Transformer(
            d_model=E,
            nhead=config.model.nb_heads,
            num_encoder_layers=config.model.nb_layers,
            num_decoder_layers=config.model.nb_layers,
            dim_feedforward=config.model.feedforward_dim,
            dropout=config.model.dropout,
            batch_first=True
        )
        self.output_proj = nn.Linear(E, V, bias=False)
        self.output_proj.weight = self.token_emb.weight

        # Loss: simple cross-entropy ignoring pad tokens.
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token, label_smoothing=self.config.model.label_smoothing)
        
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def corrupt_tokens(self, token_ids):
        """
        Corrupt tokens by randomly replacing non-pad tokens with a random token from the vocabulary.
        """
        if not self.training:
            return token_ids

        B, L = token_ids.shape
        device = token_ids.device
        V = self.config.data.vocabulary_size
        pad = self.pad_token

        # 1) sample a corruption rate per example
        alphas = torch.rand(B, device=device).unsqueeze(1)   # (B,1), each in [0,1)
        # 2) for each token generate U(0,1) noise, compare to alpha
        noise = torch.rand(B, L, device=device)
        # mask = (noise < alpha) & (token != pad)
        mask  = noise < alphas
        mask &= (token_ids != pad)

        # 3) sample random tokens (exclude pad token)
        # Create a tensor of random values between 0 and V-1
        random_tokens = torch.randint(0, V-1, (B, L), device=device)
        # Shift values >= pad up by 1 to skip the pad token
        random_tokens = random_tokens + (random_tokens >= pad).long()

        # 4) apply
        corrupted = token_ids.clone()
        corrupted[mask] = random_tokens[mask]
        return corrupted

    def _encode(self, src_ids):
        """
        Run the Transformer encoder and return:
          enc_out ........ (B, S, E) contextual embeddings
          src_key_pad .... Bool mask, True where PAD
        """
        B, S = src_ids.shape
        src_pos = torch.arange(S, device=src_ids.device).unsqueeze(0).expand(B, -1)
        src_emb = self.dropout(self.token_emb(src_ids) + self.pos_emb(src_pos))

        src_key_pad = src_ids == self.pad_token         # (B, S)
        enc_out = self.model.encoder(                   # nn.Transformer has .encoder
            src=src_emb,
            src_key_padding_mask=src_key_pad
        )                                               # (B, S, E)

        return enc_out, src_key_pad

    def _compute_length_metrics(self, length_logits, true_downsampled_len):
        """Calculate RMSE and accuracy for length prediction"""
        pred_lengths = torch.argmax(length_logits, dim=1)
        accuracy = (pred_lengths == true_downsampled_len).float().mean()
        
        # Convert indices to actual lengths for RMSE calculation
        # Use MSE loss first, then take square root
        mse = F.mse_loss(pred_lengths.float(), true_downsampled_len.float())
        rmse = torch.sqrt(mse)
        
        return rmse, accuracy

    def forward(self, src_ids, tgt_ids, length_ids=None, encoder_output=None, src_key_padding_mask=None):
        # 1) Embed and add positional encodings
        B, S = src_ids.shape
        _, T = tgt_ids.shape
        
        tgt_pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.dropout(self.token_emb(tgt_ids) + self.pos_emb(tgt_pos))

        # If encoder output is not provided, compute it
        if encoder_output is None or src_key_padding_mask is None:
            src_pos = torch.arange(S, device=src_ids.device).unsqueeze(0).expand(B, -1)
            src_emb = self.dropout(self.token_emb(src_ids) + self.pos_emb(src_pos))
            
            # Build key_padding_mask: True at padding positions
            src_key_pad = src_ids == self.pad_token
            
            # prepend length embedding if provided
            if length_ids is not None:
                # length_ids: (B,) in [0..Dd-1]
                len_e = self.length_emb(length_ids)           # (B, E)
                len_e = len_e.unsqueeze(1)                    # (B, 1, E)
                # drop last position so shapes match
                src_emb = torch.cat([len_e, src_emb[:, :-1, :]], dim=1)
                
            # Call encoder
            memory = self.model.encoder(
                src=src_emb,
                src_key_padding_mask=src_key_pad
            )
        else:
            # Use provided encoder output
            memory = encoder_output
            src_key_pad = src_key_padding_mask
            
            # prepend length embedding if provided
            if length_ids is not None:
                # length_ids: (B,) in [0..Dd-1]
                len_e = self.length_emb(length_ids)           # (B, E)
                len_e = len_e.unsqueeze(1)                    # (B, 1, E)
                # drop last position so we don't exceed the expected sequence length
                memory = torch.cat([len_e, memory[:, :-1, :]], dim=1)

        # 2) Build tgt key_padding_mask
        tgt_key_pad = tgt_ids == self.pad_token

        # 3) Call transformer decoder (no causal mask → bidirectional self-attention)
        out = self.model.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_key_pad,
            memory_key_padding_mask=src_key_pad,
            # do NOT pass any `tgt_mask` → no triangular masking
        )  # → shape (B, T, E)

        # 4) Project to logits
        logits = self.output_proj(out)  # (B, T, V)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training with unrolled denoising:
          1. Corrupt the entire target sequence.
          2. Compute loss from a forward pass with corrupted tokens (loss1).
          3. Compute predictions from the first pass via argmax (detached).
          4. Run a second forward pass using these predictions (loss2).
          5. Average the losses.
        """
        src = batch['source']
        tgt = batch['target']

        # compute true downsampled target length
        # count non-pad tokens, then ceil-divide by 2
        true_len = (tgt != self.pad_token).sum(dim=1)           # (B,)
        down_len = ((true_len + 1) // 2).clamp(max=self.config.model.downsampled_target_length - 1)

        # SINGLE encoder pass to be reused for all operations
        enc_out, src_key_pad = self._encode(src)
        
        # Length prediction using the encoded output
        with torch.no_grad():
            mask = (~src_key_pad).unsqueeze(-1)
            pooled = enc_out.masked_fill(~mask, 0.0).sum(1) / mask.sum(1).clamp(min=1)
        length_logits = self.length_pred(pooled.detach())
        length_loss  = F.cross_entropy(length_logits, down_len)
        
        # Calculate length prediction metrics
        length_rmse, length_accuracy = self._compute_length_metrics(length_logits, down_len)

        # First step: corrupt the target tokens.
        corrupted_tgt = self.corrupt_tokens(tgt)
        
        logits1 = self.forward(
            src, 
            corrupted_tgt, 
            length_ids=down_len, 
            encoder_output=enc_out, 
            src_key_padding_mask=src_key_pad
        )
        vocab_size = logits1.shape[-1]
        loss1 = self.criterion(logits1.view(-1, vocab_size), tgt.view(-1))
        
        # Detach argmax predictions from the first pass.
        pred1 = torch.argmax(logits1, dim=-1).detach()
        
        logits2 = self.forward(
            src, 
            pred1, 
            length_ids=down_len, 
            encoder_output=enc_out, 
            src_key_padding_mask=src_key_pad
        )
        loss2 = self.criterion(logits2.view(-1, vocab_size), tgt.view(-1))
        
        total_token_loss = (loss1 + loss2) / 2.0
        total_loss = total_token_loss + self.length_loss_weight * length_loss
        self.log("train_token_loss", total_token_loss, prog_bar=True, on_step=True)
        self.log("train_length_loss", length_loss, prog_bar=False, on_step=True)
        self.log("train_loss", total_loss, prog_bar=True, on_step=True)
        
        # Log length prediction metrics
        self.log("train_rmse_length_error", length_rmse, prog_bar=True, on_step=True)
        self.log("train_length_class_accuracy", length_accuracy, prog_bar=True, on_step=True)
        
        # Log individual losses
        self.log("train_loss_step1", loss1, prog_bar=False, on_step=True)
        self.log("train_loss_step2", loss2, prog_bar=False, on_step=True)
        
        # Log current learning rate
        if self.trainer is not None and hasattr(self.trainer, 'optimizers'):
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, prog_bar=True)
            
        return total_loss

    def _predict_length(self, src):
        """
        Helper to go from src IDs → downsampled length IDs.
        Mirrors the code in training_step for length_pred.
        """
        with torch.no_grad():
            enc_out, src_key_pad = self._encode(src)
            mask = (~src_key_pad).unsqueeze(-1)
            enc_out = enc_out.masked_fill(~mask, 0.0)
            pooled = enc_out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        length_logits = self.length_pred(pooled)      # (B,Dd)
        length_ids    = length_logits.argmax(dim=1)   # (B,)
        return length_ids, length_logits

    def validation_step(self, batch, batch_idx):
        """
        Standard validation using teacher forcing (no corruption).
        """
        src = batch['source']
        tgt = batch['target']
        
        # Compute true downsampled target length
        true_len = (tgt != self.pad_token).sum(dim=1)
        down_len = ((true_len + 1) // 2).clamp(max=self.config.model.downsampled_target_length - 1)
        
        # Get length predictions
        length_ids, length_logits = self._predict_length(src)
        
        # Calculate length prediction metrics
        length_rmse, length_accuracy = self._compute_length_metrics(length_logits, down_len)
        
        # Log length prediction metrics
        self.log("val_rmse_length_error", length_rmse, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val_length_class_accuracy", length_accuracy, prog_bar=True, on_epoch=True, sync_dist=True)
        
        # Run encoder once
        enc_out, src_key_pad = self._encode(src)
        
        # Get output logits
        logits = self.forward(
            src, 
            tgt, 
            length_ids=length_ids, 
            encoder_output=enc_out, 
            src_key_padding_mask=src_key_pad
        )
        vocab_size = logits.shape[-1]
        loss = self.criterion(logits.view(-1, vocab_size), tgt.view(-1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
        out = {"loss": loss, "length_rmse": length_rmse, "length_accuracy": length_accuracy}
        # Only sample from the first batch to keep validation fast
        if batch_idx == 0:
            generated_tokens = self.generate(src)
            out["generated_tokens"] = generated_tokens
            out["reference_tokens"] = tgt
        
        self.my_val_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        # All tokenization and BLEU calculation is handled here
        all_generated = []
        all_references = []
        
        # Only process outputs that contain sample translations
        for output in self.my_val_outputs:
            if "generated_tokens" in output:
                for gen_tokens, ref_tokens in zip(output["generated_tokens"], output["reference_tokens"]):
                    gen_text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
                    ref_text = self.tokenizer.decode(ref_tokens.tolist(), skip_special_tokens=True)
                    all_generated.append(gen_text)
                    all_references.append(ref_text)
        
        if all_generated and all_references:
            # For SacreBLEU, references should be a list where each item is a list of references
            # For single reference per source, we need [[ref1], [ref2], ...] format
            references_for_sacrebleu = [[ref] for ref in all_references]
            sacrebleu_score = sacrebleu.corpus_bleu(all_generated, references_for_sacrebleu)
            
            # For NLTK BLEU, tokenize and structure references correctly
            tokenized_generated = [gen.split() for gen in all_generated]
            tokenized_references = [[ref.split()] for ref in all_references]
            
            # Add smoothing function to handle zero counts
            smoothing = nltk.translate.bleu_score.SmoothingFunction()
            nltk_bleu = nltk.translate.bleu_score.corpus_bleu(
                tokenized_references, 
                tokenized_generated,
                smoothing_function=smoothing.method1
            ) * 100
            
            self.log('val_sacrebleu', sacrebleu_score.score, sync_dist=True)
            self.log('val_nltk_bleu', nltk_bleu, sync_dist=True)
            logger.info(f"Validation SacreBLEU: {sacrebleu_score.score:.2f}, NLTK BLEU: {nltk_bleu:.2f}")
            
            # Log some example translations
            num_examples = min(3, len(all_generated))
            for i in range(num_examples):
                logger.info(f"Example {i+1}:")
                logger.info(f"  Reference: {all_references[i]}")
                logger.info(f"  Generated: {all_generated[i]}")
        
        self.my_val_outputs.clear()

    def on_before_optimizer_step(self, optimizer):
        # Track gradient norms
        all_params = list(self.token_emb.parameters()) + list(self.model.parameters())
        grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in all_params if p.grad is not None]), 2)
        self.log("grad_norm", grad_norm, on_step=True, prog_bar=True)

    
    def generate(self, src, num_steps=None, temperature=None, num_samples=None):
        """
        Generate translations via iterative refinement (unrolled denoising) with multiple samples.
        
        Args:
            src (Tensor): Source token IDs, shape (batch_size, src_seq_len).
            num_steps (int, optional): Number of refinement iterations.
                If None, uses config.sample.steps.
            temperature (float, optional): Temperature scaling for logits.
                If None, uses config.sample.temperature.
            num_samples (int, optional): Number of samples to generate and rerank.
                If None, uses config.sample.num_samples.
        
        Returns:
            Tensor with generated token IDs, shape (batch_size, target_sequence_length).
        """
        batch_size, src_len = src.size()
        tgt_len = self.config.data.target_sequence_length
        V = self.config.data.vocabulary_size
        device = src.device

        # defaults
        num_steps   = num_steps   or self.config.sample.steps
        temperature = temperature or self.config.sample.temperature
        num_samples = num_samples or self.config.sample.num_samples  # e.g. 16

        # 1) predict lengths once per example
        length_ids, _ = self._predict_length(src)  # (B,)

        # 2) expand src and length_ids to (B * n)
        src_exp = src.unsqueeze(1).expand(-1, num_samples, -1)         # (B, n, S)
        src_flat = src_exp.reshape(batch_size * num_samples, src_len)  # (B*n, S)

        len_exp = length_ids.unsqueeze(1).expand(-1, num_samples)      # (B, n)
        len_flat = len_exp.reshape(batch_size * num_samples)           # (B*n,)

        # 3) initialize random targets: (B*n, Tgt)
        raw = torch.randint(V-1, (batch_size * num_samples, tgt_len), device=device)
        tgt_flat = raw + (raw >= self.pad_token).long()

        # Compute encoder outputs once and reuse them for all refinement steps
        enc_out, src_key_pad = self._encode(src_flat)

        # 4) iterative refinement on the flattened batch
        for step in range(num_steps):
            logits = self.forward(
                src_flat, 
                tgt_flat, 
                length_ids=len_flat,
                encoder_output=enc_out,
                src_key_padding_mask=src_key_pad
            )  # (B*n, T, V)
            
            if temperature < 1e-2:
                # near‐deterministic
                tgt_flat = logits.argmax(dim=-1)
            else:
                # sample at temperature
                tgt_flat = Categorical(logits=logits / temperature).sample()

        # 5) rerank by model log‑prob
        #   compute log‐softmax once more
        final_logits = self.forward(
            src_flat, 
            tgt_flat, 
            length_ids=len_flat,
            encoder_output=enc_out,
            src_key_padding_mask=src_key_pad
        )
        logp = F.log_softmax(final_logits, dim=-1)                     # (B*n, T, V)

        # gather log‐probs at the chosen tokens
        # tgt_flat.unsqueeze(-1) → (B*n, T, 1)
        tok_logp = logp.gather(-1, tgt_flat.unsqueeze(-1)).squeeze(-1)  # (B*n, T)
        seq_logp = tok_logp.sum(dim=-1)                                # (B*n,)

        # reshape back to (B, n) and pick best idx per example
        seq_logp = seq_logp.view(batch_size, num_samples)              # (B, n)
        best_idx = seq_logp.argmax(dim=1)                              # (B,)

        # reshape tgt_flat likewise and index
        tgt_reshaped = tgt_flat.view(batch_size, num_samples, tgt_len) # (B, n, T)
        best = tgt_reshaped[torch.arange(batch_size), best_idx]        # (B, T)

        logger.info(f"Generated {num_samples} samples per input and selected best after {num_steps} refinement steps.")
        
        return best

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
            lr=self.config.model.peak_lr, 
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: self._get_lr_multiplier(step)
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return [optimizer], [scheduler]
    
    def _get_lr_multiplier(self, step):
        """Custom learning rate schedule with warmup and cosine decay."""
        warmup_steps = self.config.model.warmup_steps
        max_steps = self.config.model.trainer.max_steps
        min_lr = self.config.model.min_lr
        peak_lr = self.config.model.peak_lr
        final_lr = self.config.optimizer.learning_rate

        min_multiplier = min_lr / peak_lr  # e.g., 1e-7/1e-4 = 0.001
        peak_multiplier = 1.0
        final_multiplier = final_lr / peak_lr  # e.g., 1e-5/1e-4 = 0.1
        
        if step < warmup_steps:
            return min_multiplier + (peak_multiplier - min_multiplier) * (step / warmup_steps)
        
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return final_multiplier + (peak_multiplier - final_multiplier) * cosine_decay
