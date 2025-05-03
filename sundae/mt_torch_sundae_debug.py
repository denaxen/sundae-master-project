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
        
        V = config.data.vocabulary_size
        E = config.model.embedding_dim
        N = max(config.data.source_sequence_length, config.data.target_sequence_length)

        # token + position embeddings (shared)
        self.token_emb   = nn.Embedding(V, E)
        self.token_emb.weight.data[self.pad_token].zero_() # keep value but allow grads
        self.pos_emb     = nn.Embedding(N, E)
        self.dropout     = nn.Dropout(config.model.dropout)

        self.model = nn.Transformer(
            d_model=E,
            nhead=config.model.nb_heads,
            num_encoder_layers=config.model.nb_layers,
            num_decoder_layers=config.model.nb_layers,
            dim_feedforward=config.model.feedforward_dim,
            dropout=config.model.dropout,
            batch_first=True,
            norm_first=True
        )
        self.output_proj = nn.Linear(E, V, bias=False)
        self.output_proj.weight = self.token_emb.weight
        

        # Loss: simple cross-entropy ignoring pad tokens.
        if self.config.model.ignore_pad_token:
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_token, label_smoothing=self.config.model.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.model.label_smoothing)
        
        logger.info(f"Total trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def corrupt_tokens(self, token_ids):
        if not self.training:
            return token_ids

        B, L = token_ids.shape
        device = token_ids.device
        V = self.config.data.vocabulary_size
        pad = self.pad_token

        alphas = torch.rand(B, device=device).unsqueeze(1)
        noise = torch.rand(B, L, device=device)
        mask = (noise < alphas) & (token_ids != pad)

        random_tokens = torch.randint(0, V-1, (B, L), device=device)
        random_tokens += (random_tokens >= pad).long()

        corrupted = token_ids.clone()
        corrupted[mask] = random_tokens[mask]
        return corrupted

    def _encode(self, src_ids):
        B, S = src_ids.shape
        src_key_pad = (src_ids == self.pad_token)
        src_pos = torch.arange(S, device=src_ids.device).unsqueeze(0).expand(B, -1)
        src_emb = self.dropout(self.token_emb(src_ids) + self.pos_emb(src_pos))
        enc_out = self.model.encoder(
            src=src_emb,
            src_key_padding_mask=src_key_pad
        )
        return enc_out, src_key_pad

    def forward(self, src_ids, tgt_ids, encoder_output=None, src_key_padding_mask=None):
        B, T = tgt_ids.shape
        
        tgt_pos = torch.arange(T, device=tgt_ids.device).unsqueeze(0).expand(B, -1)
        tgt_emb = self.dropout(self.token_emb(tgt_ids) + self.pos_emb(tgt_pos))

        if encoder_output is None or src_key_padding_mask is None:
            memory, src_key_pad = self._encode(src_ids)
        else:
            memory, src_key_pad = encoder_output, src_key_padding_mask

        tgt_key_pad = (tgt_ids == self.pad_token)

        out = self.model.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_key_pad,
            memory_key_padding_mask=src_key_pad,
        )
        logits = self.output_proj(out)
        return logits

    def training_step(self, batch, batch_idx):
        src = batch['source']
        tgt = batch['target']

        # Single encoder pass for token loss
        enc_out, src_key_pad = self._encode(src)
        current_tgt = self.corrupt_tokens(tgt)

        step_losses = []
        nonpad_mask = (tgt != self.pad_token)
        for step in range(self.config.unroll_steps):
            logits = self.forward(
                src, 
                current_tgt, 
                encoder_output=enc_out,
                src_key_padding_mask=src_key_pad
            )
            vocab_size = logits.shape[-1]
            loss = self.criterion(logits.view(-1, vocab_size), tgt.view(-1))
            step_losses.append(loss)
            
            if step < self.config.unroll_steps - 1:
                with torch.no_grad():
                    sampled = logits.argmax(dim=-1)
                sampled[~nonpad_mask] = self.pad_token
                current_tgt = sampled
        
        total_token_loss = sum(step_losses) / len(step_losses)
        
        self.log("train_token_loss", total_token_loss, prog_bar=True, on_step=True)
        self.log("train_loss", total_token_loss, prog_bar=True, on_step=True)
        
        # Log current learning rate
        if self.trainer is not None and hasattr(self.trainer, 'optimizers'):
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, prog_bar=True)
            
        return total_token_loss

    def validation_step(self, batch, batch_idx):
        src = batch['source']
        tgt = batch['target']
        enc_out, src_key_pad = self._encode(src)
        
        logits = self.forward(
            src,
            tgt,
            encoder_output=enc_out,
            src_key_padding_mask=src_key_pad
        )
        vocab_size = logits.shape[-1]
        loss = self.criterion(logits.view(-1, vocab_size), tgt.view(-1))
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log('val_perplexity', torch.exp(loss), prog_bar=True, sync_dist=True)
        
        generated_tokens = self.generate(src)
        out = {"generated_tokens": generated_tokens, "reference_tokens": tgt, "source": src}
        self.my_val_outputs.append(out)
        return loss

    def on_validation_epoch_end(self):
        # All tokenization and BLEU calculation is handled here
        all_generated = []
        all_gen_tokens = []
        all_references = []
        all_ref_tokens = []
        all_src_tokens = []
        # Only process outputs that contain sample translations
        for output in self.my_val_outputs:
            if "generated_tokens" in output:
                for gen_tokens, ref_tokens, src in zip(output["generated_tokens"], output["reference_tokens"], output["source"]):
                    gen_text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
                    ref_text = self.tokenizer.decode(ref_tokens.tolist(), skip_special_tokens=True)
                    all_generated.append(gen_text)
                    all_references.append(ref_text)
                    all_gen_tokens.append(gen_tokens)
                    all_ref_tokens.append(ref_tokens)
                    all_src_tokens.append(src)
        if all_generated and all_references:
            # For SacreBLEU, references should be a list where each item is a list of references
            # For single reference per source, we need [[ref1], [ref2], ...] format
            references_for_sacrebleu = [[ref] for ref in all_references]
            sacrebleu_score = sacrebleu.corpus_bleu(all_generated,
                references_for_sacrebleu,
                smooth_method='floor',
                # lowercase=True
            )
            
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
                logger.info(f"  Source: {self.tokenizer.decode(all_src_tokens[i].tolist(), skip_special_tokens=True)}")
                logger.info(f"  Reference: {all_references[i]}")
                logger.info(f"  Generated: {all_generated[i]}")
                logger.info(f"  Source Tokens: {all_src_tokens[i]}")
                logger.info(f"  Reference Tokens: {all_ref_tokens[i]}")
                logger.info(f"  Generated Tokens: {all_gen_tokens[i]}")
        logger.info("END" + "="*100)
        
        self.my_val_outputs.clear()

    def on_before_optimizer_step(self, optimizer):
        # Compute gradient norms for encoder parameters
        encoder_params = list(self.model.encoder.parameters())
        encoder_grad_norms = [p.grad.norm(2) for p in encoder_params if p.grad is not None]
        encoder_norm = torch.stack(encoder_grad_norms).norm(2) if encoder_grad_norms else torch.tensor(0.0)

        # Compute gradient norms for decoder parameters
        decoder_params = list(self.model.decoder.parameters())
        decoder_grad_norms = [p.grad.norm(2) for p in decoder_params if p.grad is not None]
        decoder_norm = torch.stack(decoder_grad_norms).norm(2) if decoder_grad_norms else torch.tensor(0.0)

        # pad_params = list(self.token_emb.weight[self.pad_token].parameters())

        self.log("encoder_grad_norm", encoder_norm, on_step=True, prog_bar=True)
        self.log("decoder_grad_norm", decoder_norm, on_step=True, prog_bar=True)
        # self.log('pad_w_norm', self.token_emb.weight[self.pad_token].norm(), on_step=True, prog_bar=True)

    @torch.no_grad()
    def generate(self, src, num_steps=None, temperature=None, num_samples=None):
        batch_size, src_len = src.size()
        tgt_len = self.config.data.target_sequence_length
        V = self.config.data.vocabulary_size
        device = src.device

        num_steps   = num_steps   or self.config.sample.steps
        temperature = temperature or self.config.sample.temperature
        num_samples = num_samples or self.config.sample.num_samples

        enc_out, src_key_pad = self._encode(src)

        best_seqs = None
        best_scores = torch.full((batch_size,), float('-inf'), device=device)

        for _ in range(num_samples):
            tgt = torch.randint(0, V-1, (batch_size, tgt_len), device=device)
            tgt = tgt + (tgt >= self.pad_token).long()
            length_mask = (torch.arange(tgt_len, device=device).unsqueeze(0) >= tgt_len)
            
            for _ in range(num_steps):
                logits = self.forward(
                    src, tgt,
                    encoder_output=enc_out,
                    src_key_padding_mask=src_key_pad
                )
                
                if temperature < 1e-2:
                    tgt = logits.argmax(dim=-1)
                else:
                    tgt = Categorical(logits=logits / temperature).sample()
                tgt = tgt.masked_fill(length_mask, self.pad_token)

            final_logits = self.forward(
                src, tgt,
                encoder_output=enc_out,
                src_key_padding_mask=src_key_pad
            )
            logp = F.log_softmax(final_logits, dim=-1)
            tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

            seq_score = (tok_logp * (~length_mask)).sum(dim=1)

            if best_seqs is None:
                best_seqs, best_scores = tgt, seq_score
            else:
                better = seq_score > best_scores
                best_scores[better] = seq_score[better]
                best_seqs[better]   = tgt[better]

        return best_seqs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
            lr=self.config.optimizer.learning_rate, 
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay)
        
        return optimizer