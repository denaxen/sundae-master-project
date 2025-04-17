import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from transformers import EncoderDecoderModel, EncoderDecoderConfig, BertConfig, AutoTokenizer
import math
from loguru import logger
import sacrebleu
import nltk
from torch.distributions.categorical import Categorical

class SundaeModel(L.LightningModule):
    """
    SUNDAE for Machine Translation implemented in PyTorch Lightning.
    
    This model builds an encoder–decoder transformer from scratch using Hugging Face's
    BertConfig and EncoderDecoderModel. It disables autoregressive masking in the decoder
    (making it bidirectional) and performs iterative denoising unrolling during training.
    
    The configuration `config` is expected to have attributes:
      - config.data.vocabulary_size, config.data.pad_token, config.data.shared_tokenizer_path
      - config.data.source_sequence_length, config.data.target_sequence_length
      - config.model.embedding_dim, nb_layers, nb_heads, feedforward_dim, dropout,
        tie_token_emb, corruption_prob, unroll_steps
      - config.optimizer.learning_rate, betas, eps, and optionally weight_decay
      - config.sample.temperature (for iterative refinement generation)
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.pad_token = config.data.pad_token
        self.eos_token = getattr(config.data, 'eos_token', None)
        # Load the shared tokenizer using the provided path.
        self.tokenizer = AutoTokenizer.from_pretrained(config.data.shared_tokenizer_path)
        
        # For validation outputs collection
        self.my_val_outputs = []
        
        # Determine max position embeddings from source and target lengths.
        max_position = max(config.data.source_sequence_length, config.data.target_sequence_length)
        
        # Create encoder and decoder configurations using BertConfig.
        encoder_config = BertConfig(
            vocab_size=config.data.vocabulary_size,
            hidden_size=config.model.embedding_dim,
            num_hidden_layers=config.model.nb_layers,
            num_attention_heads=config.model.nb_heads,
            intermediate_size=config.model.feedforward_dim,
            hidden_dropout_prob=config.model.dropout,
            attention_probs_dropout_prob=config.model.dropout,
            max_position_embeddings=max_position,
            pad_token_id=self.pad_token,
        )
        decoder_config = BertConfig(
            vocab_size=config.data.vocabulary_size,
            hidden_size=config.model.embedding_dim,
            num_hidden_layers=config.model.nb_layers,
            num_attention_heads=config.model.nb_heads,
            intermediate_size=config.model.feedforward_dim,
            hidden_dropout_prob=config.model.dropout,
            attention_probs_dropout_prob=config.model.dropout,
            max_position_embeddings=max_position,
            pad_token_id=self.pad_token,
            is_decoder=True,       # Mark as decoder...
            add_cross_attention=True,  # ...and enable cross-attention.
        )
        
        # Combine into an encoder-decoder configuration.
        ed_config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
        ed_config.tie_word_embeddings = True
        self.model = EncoderDecoderModel(ed_config)
        
        
        # Optionally tie encoder and decoder embeddings.
        enc_emb = self.model.encoder.get_input_embeddings()
        self.model.decoder.set_input_embeddings(enc_emb)
        self.model.tie_weights()
        
        # The token corruption probability (for replacing tokens randomly).
        self.corruption_prob = config.model.corruption_prob if hasattr(config.model, 'corruption_prob') else 0.3
        # Number of unrolled denoising steps (typically 2 for SUNDAE).
        self.unroll_steps = config.model.unroll_steps if hasattr(config.model, 'unroll_steps') else 2

        # Loss: simple cross-entropy ignoring pad tokens.
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token)
        
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def corrupt_tokens(self, token_ids):
        """
        Corrupt tokens by randomly replacing non-pad tokens with a random token from the vocabulary.
        """
        if not self.training:
            return token_ids
        # Create a corruption mask.
        noise = torch.rand(token_ids.shape, device=token_ids.device)
        mask = (noise < self.corruption_prob) & (token_ids != self.pad_token)
        # Random tokens uniformly sampled.
        random_tokens = torch.randint(low=0, high=self.config.data.vocabulary_size, size=token_ids.shape, device=token_ids.device)
        corrupted = token_ids.clone()
        corrupted[mask] = random_tokens[mask]
        return corrupted

    def forward(self, src_ids, src_mask, decoder_input_ids):
        """
        Forward pass through the encoder–decoder model.
        """
        outputs = self.model(
            input_ids=src_ids,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False
        )
        return outputs.logits

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
        src_mask = (src != self.pad_token).long()

        # First step: corrupt the target tokens.
        corrupted_tgt = self.corrupt_tokens(tgt)
        logits1 = self.forward(src, src_mask, corrupted_tgt)
        vocab_size = logits1.shape[-1]
        loss1 = self.criterion(logits1.view(-1, vocab_size), tgt.view(-1))
        
        # Detach argmax predictions from the first pass.
        pred1 = torch.argmax(logits1, dim=-1).detach()
        
        # Second step: feed first-step predictions.
        logits2 = self.forward(src, src_mask, pred1)
        loss2 = self.criterion(logits2.view(-1, vocab_size), tgt.view(-1))
        
        total_loss = (loss1 + loss2) / 2.0
        self.log("train_token_loss", total_loss, prog_bar=True, on_step=True)
        
        # Log individual losses
        self.log("train_loss_step1", loss1, prog_bar=False, on_step=True)
        self.log("train_loss_step2", loss2, prog_bar=False, on_step=True)
        
        # Log current learning rate
        if self.trainer is not None and hasattr(self.trainer, 'optimizers'):
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, prog_bar=True)
            
        return total_loss

    def sample_translation(self, src, min_steps=4):
        """Generate translation for a given source batch.
        
        This function works similarly to generate() but follows the implementation style
        from the original models_mt_sundae.py.
        
        Args:
            src (Tensor): Source token IDs, shape (batch_size, src_seq_len).
            min_steps (int): Minimum number of refinement iterations.
            
        Returns:
            Tensor with generated token IDs, shape (batch_size, target_sequence_length).
        """
        src_mask = (src != self.pad_token).long()
        batch_size = src.size(0)
        max_len = self.config.data.target_sequence_length
        device = src.device
        
        # Start with random tokens (from uniform prior) as initial target
        init_tgt = torch.randint(
            self.config.data.vocabulary_size,
            (batch_size, max_len),
            device=device
        )
        
        # Get configured values for sampling
        num_steps = getattr(self.config.sample, 'steps', 10)
        temperature = getattr(self.config.sample, 'temperature', 1.0)
        
        # Use min_steps as a lower bound for the number of steps
        num_steps = max(num_steps, min_steps)
        
        for step_idx in range(num_steps):
            logits = self.forward(src, src_mask, init_tgt)
            
            # Apply temperature scaling
            if temperature < 0.01:
                # If temperature is very low, use deterministic argmax
                sample = torch.argmax(logits, dim=-1)
            else:
                # Otherwise sample from scaled logits
                sample = Categorical(logits=logits / temperature).sample()
                
            init_tgt = sample
            
        logger.info(f"Stopped sampling after {step_idx+1} steps.")
        
        # Optionally trim sequences based on EOS token
        if self.eos_token is not None and getattr(self.config.sample, 'trim_eos', False):
            # Find first EOS token in each sequence and trim
            eos_positions = (init_tgt == self.eos_token).nonzero()
            for i in range(batch_size):
                # Get positions where EOS appears in sequence i
                seq_eos = eos_positions[eos_positions[:, 0] == i]
                if len(seq_eos) > 0:
                    # Take first EOS position and trim sequence
                    first_eos = seq_eos[0, 1]
                    init_tgt[i, first_eos+1:] = self.pad_token
        
        return init_tgt

    def validation_step(self, batch, batch_idx):
        """
        Standard validation using teacher forcing (no corruption).
        """
        src = batch['source']
        tgt = batch['target']
        src_mask = (src != self.pad_token).long()
        logits = self.forward(src, src_mask, tgt)
        vocab_size = logits.shape[-1]
        loss = self.criterion(logits.view(-1, vocab_size), tgt.view(-1))
        self.log("val_loss", loss, prog_bar=True, on_step=False, sync_dist=True)
        
        out = {"loss": loss}
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
        encoder_params = list(self.model.encoder.parameters())
        encoder_grad_norms = [p.grad.norm(2) for p in encoder_params if p.grad is not None]
        encoder_norm = torch.stack(encoder_grad_norms).norm(2) if encoder_grad_norms else torch.tensor(0.0)

        decoder_params = list(self.model.decoder.parameters())
        decoder_grad_norms = [p.grad.norm(2) for p in decoder_params if p.grad is not None]
        decoder_norm = torch.stack(decoder_grad_norms).norm(2) if decoder_grad_norms else torch.tensor(0.0)

        self.log("encoder_grad_norm", encoder_norm, on_step=True, prog_bar=True)
        self.log("decoder_grad_norm", decoder_norm, on_step=True, prog_bar=True)

    def generate(self, src, num_steps=None, temperature=None):
        """
        Generate translations via iterative refinement (unrolled denoising).
        
        Args:
            src (Tensor): Source token IDs, shape (batch_size, src_seq_len).
            num_steps (int, optional): Number of refinement iterations.
                If None, uses config.sample.steps.
            temperature (float, optional): Temperature scaling for logits.
                If None, uses config.sample.temperature.
        
        Returns:
            Tensor with generated token IDs, shape (batch_size, target_sequence_length).
        """
        src_mask = (src != self.pad_token).long()
        batch_size = src.size(0)
        tgt_len = self.config.data.target_sequence_length
        device = src.device
        
        # Use configured values if not provided
        if num_steps is None:
            num_steps = getattr(self.config.sample, 'steps', 10)
        if temperature is None:
            temperature = getattr(self.config.sample, 'temperature', 1.0)
        
        # Start with random tokens (from uniform prior) as initial target.
        current_tgt = torch.randint(low=0, high=self.config.data.vocabulary_size, size=(batch_size, tgt_len), device=device)
        
        for step_idx in range(num_steps):
            logits = self.forward(src, src_mask, current_tgt)
            
            # Apply temperature scaling
            # If temperature is very low, use argmax; otherwise sample from the distribution
            if temperature < 0.01:
                current_tgt = torch.argmax(logits, dim=-1)
            else:
                current_tgt = Categorical(logits=logits / temperature).sample()
                
        logger.info(f"Stopped sampling after {num_steps} steps.")
        
        # Optionally trim based on EOS token
        if self.eos_token is not None and getattr(self.config.sample, 'trim_eos', False):
            # Find first EOS token in each sequence and trim
            eos_positions = (current_tgt == self.eos_token).nonzero()
            for i in range(batch_size):
                # Get positions where EOS appears in sequence i
                seq_eos = eos_positions[eos_positions[:, 0] == i]
                if len(seq_eos) > 0:
                    # Take first EOS position and trim sequence
                    first_eos = seq_eos[0, 1]
                    current_tgt[i, first_eos+1:] = self.pad_token
        
        return current_tgt

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
