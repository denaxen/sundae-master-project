import math
import torch
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import lightning as L
from x_transformers import TransformerWrapper, Encoder, Decoder
from loguru import logger
import sacrebleu
import nltk
import math

class SundaeMTModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.my_val_outputs = []
        
        # Encoder: process the source sentence
        self.encoder = TransformerWrapper(
            num_tokens=config.data.vocabulary_size,
            max_seq_len=config.data.source_sequence_length,
            emb_dropout=config.model.dropout,
            attn_layers=Encoder(
                dim=config.model.embedding_dim,
                depth=config.model.nb_layers,
                heads=config.model.nb_heads,
                ff_mult=config.model.feedforward_dim // config.model.embedding_dim,
                attn_dropout=config.model.dropout,
                ff_dropout=config.model.dropout,
                layer_dropout=config.model.dropout,
            ),
            return_only_embed=True
        )
        
        # Decoder: generate target sentence (non-autoregressive, no causal mask)
        self.decoder = TransformerWrapper(
            num_tokens=config.data.vocabulary_size,
            max_seq_len=config.data.target_sequence_length,
            emb_dropout=config.model.dropout,
            # using encoder-type layers with cross-attention for conditioning
            attn_layers=Encoder(
                dim=config.model.embedding_dim,
                depth=config.model.nb_layers,   
                heads=config.model.nb_heads,
                ff_mult=config.model.feedforward_dim // config.model.embedding_dim,
                cross_attend=True,
                attn_dropout=config.model.dropout,
                ff_dropout=config.model.dropout,
                layer_dropout=config.model.dropout,
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
        # Length Predictor (detached from encoder)
        # -------------------------
        self.length_predictor = torch.nn.Sequential(
            torch.nn.Linear(config.model.embedding_dim, config.model.target_length_prediction_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(config.model.target_length_prediction_hidden_dim, config.model.downsampled_target_length)
        )

        # Share token embeddings between encoder and decoder
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

        # Detach encoder output for length prediction to avoid backprop into encoder
        with torch.no_grad():
            src_enc_detached = src_enc.clone().detach()
        
        # Predict length from the mean-pooled detached encoding
        pred_length_logits = self.length_predictor(src_enc_detached.mean(dim=1))

        # Compute ground-truth length classes (downsampled)
        gt_len = (tgt != self.config.data.pad_token).sum(dim=1)  # actual token count per sentence
        gt_len_downsampled = torch.clamp((gt_len + 1) // 2, max=self.config.model.downsampled_target_length - 1)
        length_emb = self.length_embed(gt_len_downsampled)  # shape: [batch_size, d_model]

        # Prepend length embedding to encoder output
        length_emb = length_emb.unsqueeze(1)  # shape: [batch_size, 1, d_model]
        src_enc_with_len = torch.cat([length_emb, src_enc], dim=1)
        
        # Begin with a corrupted version of the target text
        current_tgt = self.corrupt_text(tgt)
        all_logits = []
        
        # Unrolled denoising steps
        for _ in range(self.config.unroll_steps):
            logits = self.decoder(current_tgt, context=src_enc_with_len)
            current_tgt = Categorical(logits=logits).sample().detach()
            all_logits.append(logits)
        
        final_logits = torch.cat(all_logits, dim=0)
        return final_logits, pred_length_logits
    
    def training_step(self, batch, batch_idx):
        src, tgt = batch['source'], batch['target']
        logits, pred_length_logits = self.forward(src, tgt)
        
        repeated_tgt = tgt.repeat(self.config.unroll_steps, 1)
        token_loss = F.cross_entropy(logits.permute(0, 2, 1),
            repeated_tgt,
            label_smoothing=self.config.model.label_smoothing,
            ignore_index=self.config.data.pad_token)
        
        gt_len = (tgt != self.config.data.pad_token).sum(dim=1)
        gt_len_downsampled = torch.clamp((gt_len + 1) // 2, max=self.config.model.downsampled_target_length - 1)
        length_loss = F.cross_entropy(pred_length_logits, gt_len_downsampled)
        
        loss = token_loss + self.config.model.length_loss_weight * length_loss
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_token_loss', token_loss, prog_bar=True)
        self.log('train_length_loss', length_loss, prog_bar=True)
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
        
        pred_len_class = pred_length_logits.argmax(dim=-1)
        length_class_accuracy = (pred_len_class == gt_len_downsampled).float().mean()
        self.log('train_length_class_accuracy', length_class_accuracy, prog_bar=True)
        
        pred_token_len = pred_len_class * 2  # upsampled approximate token length
        mse_length_error = ((pred_token_len - gt_len) ** 2).float().mean()
        rmse_length_error = torch.sqrt(mse_length_error)
        self.log('train_rmse_length_error', rmse_length_error, prog_bar=True)
        
        mean_pred_len = pred_token_len.float().mean()
        std_pred_len = pred_token_len.float().std()
        mean_gt_len = gt_len.float().mean()
        self.log('train_mean_pred_length', mean_pred_len)
        self.log('train_std_pred_length', std_pred_len)
        self.log('train_mean_gt_length', mean_gt_len)
        
        return loss

    def sample_translation(self, src, min_steps=4):
        """Generate translation for a given source batch."""
        src_enc = self.encoder(src)
        pred_length_logits = self.length_predictor(src_enc.mean(dim=1))
        pred_len_class = pred_length_logits.argmax(dim=-1)
        
        length_emb = self.length_embed(pred_len_class)
        length_emb = length_emb.unsqueeze(1)
        src_enc_with_len = torch.cat([length_emb, src_enc], dim=1)
        
        batch_size = src.shape[0]
        max_len = self.config.data.target_sequence_length
        init_tgt = torch.randint(
            self.config.data.vocabulary_size,
            (batch_size, max_len),
            device=src.device
        )
        predicted_len_upsampled = pred_len_class * 2
        for i in range(batch_size):
            length_i = predicted_len_upsampled[i]
            if length_i < max_len:
                init_tgt[i, length_i:] = self.config.data.pad_token

        for step_idx in range(self.config.sample.steps):
            logits = self.decoder(init_tgt, context=src_enc_with_len)
            sample = Categorical(logits=logits / self.config.sample.temperature).sample()
            init_tgt = sample
        logger.info(f"Stopped sampling after {step_idx+1} steps.")
        
        if self.config.sample.get('trim_eos', False):  # Only trim if config flag is True
            # Find first EOS token in each sequence and trim
            eos_positions = (init_tgt == self.config.data.eos_token).nonzero()
            for i in range(batch_size):
                # Get positions where EOS appears in sequence i
                seq_eos = eos_positions[eos_positions[:, 0] == i]
                if len(seq_eos) > 0:
                    # Take first EOS position and trim sequence
                    first_eos = seq_eos[0, 1]
                    init_tgt[i, first_eos+1:] = self.config.data.pad_token
        
        return init_tgt
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch['source'], batch['target']
        logits, pred_length_logits = self.forward(src, tgt)
        repeated_tgt = tgt.repeat(self.config.unroll_steps, 1)
        token_loss = F.cross_entropy(logits.permute(0, 2, 1),
            repeated_tgt,
            label_smoothing=self.config.model.label_smoothing,
            ignore_index=self.config.data.pad_token)
        gt_len = (tgt != self.config.data.pad_token).sum(dim=1)
        gt_len_downsampled = torch.clamp((gt_len + 1) // 2, max=self.config.model.downsampled_target_length - 1)
        length_loss = F.cross_entropy(pred_length_logits, gt_len_downsampled)
        loss = token_loss + self.config.model.length_loss_weight * length_loss
        self.log('val_loss', loss, prog_bar=True)
        
        out = {"loss": loss}
        # Only sample from the first batch (to keep evaluation fast)
        if batch_idx == 0:
            generated_tokens = self.sample_translation(src)
            out["generated_tokens"] = generated_tokens
            out["reference_tokens"] = tgt
        self.my_val_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        # Load the tokenizer if not already loaded
        target_lang = "de" if not self.config.data.get("reverse", False) else "en"
        target_tokenizer_path = self.config.data.de_tokenizer_path if target_lang == "de" else self.config.data.en_tokenizer_path
        if not hasattr(self, 'target_tokenizer'):
            from transformers import AutoTokenizer
            self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_path)
        
        all_generated = []
        all_references = []
        # Only process outputs that contain sample translations
        for output in self.my_val_outputs:
            if "generated_tokens" in output:
                for gen_tokens, ref_tokens in zip(output["generated_tokens"], output["reference_tokens"]):
                    gen_text = self.target_tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
                    ref_text = self.target_tokenizer.decode(ref_tokens.tolist(), skip_special_tokens=True)
                    all_generated.append(gen_text)
                    all_references.append(ref_text)
        
        if all_generated and all_references:
            # Fix 1: For SacreBLEU, references should be a list where each item is a list of references
            # For single reference per source, we need [[ref1], [ref2], ...] format
            references_for_sacrebleu = [[ref] for ref in all_references]
            sacrebleu_score = sacrebleu.corpus_bleu(all_generated, references_for_sacrebleu)
            
            # Fix 2: For NLTK BLEU, we need to tokenize and structure references correctly
            tokenized_generated = [gen.split() for gen in all_generated]
            # Each reference needs to be a list in a list - [[tokens1], [tokens2], ...]
            tokenized_references = [[ref.split()] for ref in all_references]
            nltk_bleu = nltk.translate.bleu_score.corpus_bleu(tokenized_references, tokenized_generated) * 100
            
            self.log('val_sacrebleu', sacrebleu_score.score)
            self.log('val_nltk_bleu', nltk_bleu)
            logger.info(f"Validation SacreBLEU: {sacrebleu_score.score:.2f}, NLTK BLEU: {nltk_bleu:.2f}")
        
        self.my_val_outputs.clear()
    
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
