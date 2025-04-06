import math
import torch
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from transformers import BartForConditionalGeneration, BartConfig, AutoTokenizer
import sacrebleu
import nltk

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
        Also generates sample translations from the first batch.
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
        
        # Only sample translations from the first batch to save time.
        if batch_idx == 0:
            generated_tokens = self.sample_translation(src)
            out["generated_tokens"] = generated_tokens
            out["reference_tokens"] = tgt
        self.my_val_outputs.append(out)
        return out

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
        
        self.my_val_outputs.clear()

    def sample_translation(self, src, nb_samples=4):
        """
        Generate translations for a source batch using the model's generate method.
        """
        src_mask = (src != self.pad_token_id).long()
        batch_size = src.shape[0]
        # (Optional) you can also prepend a bos token if needed.
        start_tokens = torch.full((batch_size, 1), self.bos_token, dtype=torch.long, device=src.device)
        
        generated = self.model.generate(
            input_ids=src,
            attention_mask=src_mask,
            decoder_start_token_id=self.bos_token,
            max_length=self.config.data.target_sequence_length,
            do_sample=False,
            num_beams=4,
            length_penalty=self.config.sample.length_penalty,
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
        # return min(step ** (-0.5), step * (warmup_steps ** (-1.5)))
        return 0.0002
