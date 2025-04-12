import math
import torch
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from transformers import BartForConditionalGeneration, BartConfig, AutoTokenizer
import sacrebleu
import nltk
import glob
import os
from pathlib import Path
import re

class ARTransformerHF(L.LightningModule):
    """
    A Hugging Face Transformers based autoregressive encoder–decoder model
    for machine translation. This implementation uses a Bart model configured
    to mirror the original "Attention is All You Need" Transformer base architecture.
    Label smoothing is applied in the training step.
    """

    def __init__(self, config):
        """
        Args:
            config: an object or dict with the following (and additional) fields:
                - model.embedding_dim (e.g. 512)
                - model.nb_layers (e.g. 6)
                - model.nb_heads (e.g. 8)
                - model.feedforward_dim (e.g. 2048)
                - model.dropout (e.g. 0.1)
                - model.tie_token_emb (bool), whether to tie encoder/decoder embeddings
                - model.label_smoothing (e.g. 0.1)
                - model.peak_lr (peak learning rate for training)
                - model.warmup_steps
                - model.min_lr
                - model.trainer.max_steps
                - data.vocabulary_size (vocab size)
                - data.source_sequence_length (max source sequence length)
                - data.target_sequence_length (max target sequence length)
                - data.max_tgt_len (max target length for generation)
                - data.pad_token (pad token id)
                - data.bos_token (begin-of-sequence token id)
                - data.shared_tokenizer_path (path to shared tokenizer)
                - optimizer.learning_rate (final learning rate)
                - optimizer.betas, optimizer.eps, optimizer.weight_decay
                - sample.temperature (optional, for sampling in generation)
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.my_val_outputs = []  # For storing validation outputs
        self.pad_token_id = config.data.pad_token
        self.bos_token = config.data.bos_token
        self.original_state_dict = None  # To store original weights for later restoration
        
        # Build a Bart configuration matching our Transformer base design.
        hf_config = BartConfig(
            vocab_size=config.data.vocabulary_size,
            d_model=config.model.embedding_dim,
            encoder_layers=config.model.nb_layers,
            decoder_layers=config.model.nb_layers,
            encoder_attention_heads=config.model.nb_heads,
            decoder_attention_heads=config.model.nb_heads,
            encoder_ffn_dim=config.model.feedforward_dim,
            decoder_ffn_dim=config.model.feedforward_dim,
            dropout=config.model.dropout,
            attention_dropout=config.model.dropout,
            # Use the maximum of source/target lengths for position embeddings
            max_position_embeddings=max(config.data.source_sequence_length, config.data.target_sequence_length),
            tie_word_embeddings=config.model.tie_token_emb,
        )
        
        self.model = BartForConditionalGeneration(hf_config)
        
        if config.model.tie_token_emb:
            self.model.tie_weights()

    def training_step(self, batch, batch_idx):
        """
        Training step with label smoothing applied.
        The target is shifted: decoder inputs are all tokens except the last,
        and labels are all tokens except the first.
        """
        src, tgt = batch['source'], batch['target']
        # Create attention mask: 1 for tokens that are not pad.
        src_mask = (src != self.pad_token_id).long()
        
        # Shift the target: decoder inputs and labels.
        decoder_input_ids = tgt[:, :-1]
        labels = tgt[:, 1:]
        
        # Forward pass
        outputs = self.model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits  # shape: (B, T, vocab_size)
        vocab_size = logits.size(-1)
        
        # Compute cross-entropy loss with label smoothing.
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            label_smoothing=self.config.model.label_smoothing,
            ignore_index=self.pad_token_id,
        )
        
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True, sync_dist=True)
        return loss

    def on_before_optimizer_step(self, optimizer):
        # Compute gradient norms for encoder parameters
        encoder_params = list(self.model.model.encoder.parameters())
        encoder_grad_norms = [p.grad.norm(2) for p in encoder_params if p.grad is not None]
        encoder_norm = torch.stack(encoder_grad_norms).norm(2) if encoder_grad_norms else torch.tensor(0.0)

        # Compute gradient norms for decoder parameters
        decoder_params = list(self.model.model.decoder.parameters())
        decoder_grad_norms = [p.grad.norm(2) for p in decoder_params if p.grad is not None]
        decoder_norm = torch.stack(decoder_grad_norms).norm(2) if decoder_grad_norms else torch.tensor(0.0)

        self.log("encoder_grad_norm", encoder_norm, on_step=True, prog_bar=True)
        self.log("decoder_grad_norm", decoder_norm, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        """
        Validation step. Similar to training but without label smoothing.
        Also generates sample translations using beam search for BLEU calculation.
        """
        src, tgt = batch['source'], batch['target']
        src_mask = (src != self.pad_token_id).long()
        
        decoder_input_ids = tgt[:, :-1]
        labels = tgt[:, 1:]
        outputs = self.model(
            input_ids=src,
            attention_mask=src_mask,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        logits = outputs.logits
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        val_perplexity = torch.exp(loss)
        self.log('val_perplexity', val_perplexity, prog_bar=True, sync_dist=True)
        out = {"loss": loss}
        
        # Generate translations for all batches to calculate BLEU properly
        # Use beam search with parameters from config for BLEU calculation
        generated_tokens = self.sample_translation(
            src, 
            num_beams=self.config.sample.num_beams, 
            do_sample=self.config.sample.do_sample,
            length_penalty=self.config.sample.length_penalty
        )
        out["generated_tokens"] = generated_tokens
        out["reference_tokens"] = tgt
        
        self.my_val_outputs.append(out)
        return out

    def on_validation_epoch_start(self):
        """
        Load and average weights from the last N checkpoints for evaluation.
        N is controlled by config.checkpointing.num_checkpoints_to_average
        """
        # Save current model weights to restore after validation
        self.original_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        # Load and average checkpoints
        try:
            # Get the checkpoint directory from the ModelCheckpoint callback
            checkpoint_dir = None
            for callback in self.trainer.callbacks:
                if isinstance(callback, L.pytorch.callbacks.ModelCheckpoint):
                    checkpoint_dir = callback.dirpath
                    break
            
            if checkpoint_dir:
                # Get number of checkpoints to average from config
                num_checkpoints = self.config.checkpointing.num_checkpoints_to_average
                # Average model weights from last N checkpoints
                avg_state_dict = self._average_checkpoints(checkpoint_dir, num_checkpoints=num_checkpoints)
                if avg_state_dict:
                    self.model.load_state_dict(avg_state_dict)
                    print(f"Successfully loaded averaged weights from {num_checkpoints} checkpoints for evaluation")
        except Exception as e:
            print(f"Error loading averaged checkpoints: {str(e)}")
            # If something goes wrong, make sure we're using the original weights
            if self.original_state_dict:
                self.model.load_state_dict(self.original_state_dict)

    def on_validation_epoch_end(self):
        """
        At the end of validation, decode the generated and reference tokens using
        the shared tokenizer, and compute BLEU scores using sacreBLEU and NLTK.
        """
        # Load shared tokenizer if not already loaded.
        tokenizer_path = self.config.data.shared_tokenizer_path
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        all_generated = []
        all_references = []
        for output in self.my_val_outputs:
            if "generated_tokens" in output:
                for gen_tokens, ref_tokens in zip(output["generated_tokens"], output["reference_tokens"]):
                    gen_text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
                    ref_text = self.tokenizer.decode(ref_tokens.tolist(), skip_special_tokens=True)
                    all_generated.append(gen_text)
                    all_references.append(ref_text)
        
        if all_generated and all_references:
            # SacreBLEU expects a list of lists for references.
            references_for_sacrebleu = [[ref] for ref in all_references]
            sacrebleu_score = sacrebleu.corpus_bleu(all_generated, references_for_sacrebleu)
            
            # NLTK BLEU requires tokenized sentences.
            tokenized_generated = [gen.split() for gen in all_generated]
            tokenized_references = [[ref.split()] for ref in all_references]
            smoothing = nltk.translate.bleu_score.SmoothingFunction()
            nltk_bleu = nltk.translate.bleu_score.corpus_bleu(tokenized_references, tokenized_generated, smoothing_function=smoothing.method1) * 100
            
            self.log('val_sacrebleu', sacrebleu_score.score, sync_dist=True)
            self.log('val_nltk_bleu', nltk_bleu, sync_dist=True)
            print(f"Validation SacreBLEU: {sacrebleu_score.score:.2f}, NLTK BLEU: {nltk_bleu:.2f}")
        
        # Restore original weights to continue training
        if self.original_state_dict:
            self.model.load_state_dict(self.original_state_dict)
            self.original_state_dict = None
            
        self.my_val_outputs.clear()

    def _average_checkpoints(self, checkpoint_dir, num_checkpoints=10):
        """
        Average weights from the last n checkpoints.
        
        Args:
            checkpoint_dir (str): Directory where checkpoints are saved
            num_checkpoints (int): Number of most recent checkpoints to average
            
        Returns:
            dict: Averaged state dictionary or None if no checkpoints found
        """
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            print(f"Checkpoint directory {checkpoint_dir} does not exist")
            return None
            
        # Find all checkpoint files, but skip 'last.ckpt'
        checkpoint_files = [f for f in checkpoint_path.glob("*.ckpt") if f.name != "last.ckpt"]
        
        if not checkpoint_files:
            print(f"No checkpoint files found in {checkpoint_dir}")
            return None
            
        # Extract step numbers from filenames and sort by step (most recent first)
        step_pattern = re.compile(r".*step-(\d+).*\.ckpt$")
        
        def get_step(filename):
            match = step_pattern.match(str(filename))
            if match:
                return int(match.group(1))
            # If no step in filename, try epoch-step format
            epoch_step_pattern = re.compile(r".*epoch=(\d+)-step=(\d+).*\.ckpt$")
            match = epoch_step_pattern.match(str(filename))
            if match:
                return int(match.group(2))
            # For other patterns like "{epoch:02d}-{step:08d}"
            epoch_step_alt_pattern = re.compile(r".*-(\d+)\.ckpt$")
            match = epoch_step_alt_pattern.match(str(filename))
            if match:
                return int(match.group(1))
            return 0
            
        checkpoint_files_with_steps = [(f, get_step(f)) for f in checkpoint_files]
        sorted_checkpoints = sorted(checkpoint_files_with_steps, key=lambda x: x[1], reverse=True)
        
        # Take only the most recent n checkpoints
        checkpoints_to_average = sorted_checkpoints[:num_checkpoints]
        
        if not checkpoints_to_average:
            print("Could not find checkpoints with step information")
            return None
            
        print(f"Averaging {len(checkpoints_to_average)} checkpoints:")
        for ckpt, step in checkpoints_to_average:
            print(f"  - {ckpt.name} (step {step})")
            
        # Load and average the models
        avg_state_dict = {}
        for i, (ckpt_file, _) in enumerate(checkpoints_to_average):
            checkpoint = torch.load(ckpt_file, map_location=self.device)
            state_dict = checkpoint['state_dict']
            
            # Get the model state dict and fix the key names
            model_state_dict = {}
            for k, v in state_dict.items():
                # Handle three cases:
                # 1. Keys starting with 'model.model.' need to be converted to 'model.'
                # 2. Keys like 'model.final_logits_bias' need to be converted to just 'final_logits_bias'
                # 3. Keys like 'model.lm_head.weight' need to be converted to just 'lm_head.weight'
                if k.startswith('model.model.'):
                    new_key = k.replace('model.model.', 'model.', 1)
                    model_state_dict[new_key] = v
                elif k.startswith('model.'):
                    # For top-level model attributes, remove the 'model.' prefix entirely
                    if k in ['model.final_logits_bias', 'model.lm_head.weight']:
                        new_key = k.replace('model.', '', 1)
                        model_state_dict[new_key] = v
                    else:
                        model_state_dict[k] = v
            
            if i == 0:
                # Initialize with the first checkpoint
                avg_state_dict = {k: v.clone() for k, v in model_state_dict.items()}
            else:
                # Add to the running average
                for k, v in model_state_dict.items():
                    if k in avg_state_dict:
                        avg_state_dict[k] += v
                    else:
                        # Handle case where a checkpoint might have keys others don't
                        avg_state_dict[k] = v.clone()
                        
        # Divide by the number of checkpoints to get the average
        num_ckpts = len(checkpoints_to_average)
        for k in avg_state_dict:
            avg_state_dict[k] /= num_ckpts
            
        return avg_state_dict

    def sample_translation(self, src, num_beams=None, do_sample=None, length_penalty=None):
        """
        Generate translations for a source batch using the model's generate method.
        
        Args:
            src: Source token ids
            num_beams: Number of beams for beam search (default: from config)
            do_sample: Whether to use sampling (default: from config)
            length_penalty: Length penalty for beam search (default: from config)
            
        Returns:
            generated: Token ids of the generated translations
        """
        src_mask = (src != self.pad_token_id).long()
        batch_size = src.shape[0]
        
        # Use defaults from config if not specified
        if num_beams is None:
            num_beams = self.config.sample.num_beams
        if do_sample is None:
            do_sample = self.config.sample.do_sample
        if length_penalty is None:
            length_penalty = self.config.sample.length_penalty
        
        generated = self.model.generate(
            input_ids=src,
            attention_mask=src_mask,
            decoder_start_token_id=self.bos_token,
            max_length=self.config.data.target_sequence_length,
            do_sample=do_sample,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True
        )
        return generated

    def configure_optimizers(self):
        """
        Set up the optimizer and a learning rate scheduler with a warmup
        and cosine decay schedule.
        """
        d_model = self.config.model.embedding_dim
        warmup_steps = self.config.model.warmup_steps
        base_lr = d_model ** -0.5
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=base_lr,
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
            # weight_decay=self.config.optimizer.weight_decay
        )
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: self._get_inverse_sqrt_lr_multiplier(step)
            ),
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def _get_inverse_sqrt_lr_multiplier(self, step: int) -> float:
        """
        Computes the multiplier for the learning rate using the inverse square root schedule.
        
        The multiplier is:
            min(step^(-0.5), step * warmup_steps^(-1.5))
            
        We ensure step is at least 1 to avoid division by zero.
        """
        warmup_steps = self.config.model.warmup_steps
        if step < 1:
            step = 1
        return min(step ** (-0.5), step * (warmup_steps ** (-1.5)))
