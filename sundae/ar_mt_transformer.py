import torch
import torch.nn.functional as F
from torch.optim import Adam
import lightning as L
from x_transformers import XTransformer
import math
import sacrebleu
import nltk

def generate_tgt_mask(tgt, pad_token_id):
    B, T = tgt.size()
    # Create a lower-triangular (causal) mask: True means allowed.
    causal_mask = torch.tril(torch.ones((T, T), dtype=torch.bool, device=tgt.device))
    
    # Create the target padding mask: True where tokens are not pad.
    tgt_padding_mask = (tgt != pad_token_id)  # shape: (B, T)
    
    # Expand masks to combine: 
    # - causal_mask -> (1, T, T)
    # - tgt_padding_mask -> (B, 1, T)
    combined_mask = causal_mask.unsqueeze(0) & tgt_padding_mask.unsqueeze(1)
    
    # Now unsqueeze to add the head dimension: (B, 1, T, T)
    return combined_mask.unsqueeze(1)

class ARTransformerBase(L.LightningModule):
    """
    A standard Transformer 'base' autoregressive encoder-decoder model.
    Uses x_transformers' XTransformer under the hood.
    """

    def __init__(self, config):
        """
        Args:
            config: an object or dict with fields such as:
                - model.dim (e.g. 512)
                - model.num_layers (e.g. 6)
                - model.num_heads (e.g. 8)
                - model.vocab_size (e.g. 32000)
                - model.tie_token_emb (bool), whether to tie encoder/decoder embeddings
                - data.max_src_len (max length for source)
                - data.max_tgt_len (max length for target)
                - data.pad_token_id (ID of the pad token)
                - optimizer.lr (learning rate)
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.my_val_outputs = []  # Add this to store validation outputs
        
        # Create a standard Transformer with more detailed config
        self.pad_token_id = config.data.pad_token
        self.model = XTransformer(
            dim = config.model.embedding_dim,  # renamed from dim for consistency
            enc_num_tokens = config.data.vocabulary_size,  # renamed from vocab_size
            enc_depth = config.model.nb_layers,  # renamed from num_layers
            enc_heads = config.model.nb_heads,  # renamed from num_heads
            enc_max_seq_len = config.data.source_sequence_length,  # renamed from max_src_len
            dec_num_tokens = config.data.vocabulary_size,
            dec_depth = config.model.nb_layers,
            dec_heads = config.model.nb_heads,
            dec_max_seq_len = config.data.target_sequence_length,  # renamed from max_tgt_len
            
            # Add these new parameters
            emb_dropout = config.model.dropout,
            attn_dropout = config.model.dropout,
            ff_dropout = config.model.dropout,
            ff_mult = config.model.feedforward_dim // config.model.embedding_dim,
            pad_value = self.pad_token_id,
            ignore_index=self.pad_token_id
        )

        # Store a few parameters locally for convenience
        self.lr = config.optimizer.learning_rate  # use final_lr from optimizer config


    def training_step(self, batch, batch_idx):
        """
        Single step of training.
        batch is assumed to be a dict like: {'source': ..., 'target': ...}.
        """
        src, tgt = batch['source'], batch['target']
        
        # # Load tokenizers if not already loaded
        # if not hasattr(self, 'source_tokenizer') or not hasattr(self, 'target_tokenizer'):
        #     from transformers import AutoTokenizer
        #     # Determine source/target languages based on reverse flag
        #     source_lang = "de" if self.config.data.get("reverse", False) else "en"
        #     target_lang = "en" if source_lang == "de" else "de"
            
        #     source_tokenizer_path = self.config.data.de_tokenizer_path if source_lang == "de" else self.config.data.en_tokenizer_path
        #     target_tokenizer_path = self.config.data.de_tokenizer_path if target_lang == "de" else self.config.data.en_tokenizer_path
            
        #     self.source_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_path)
        #     self.target_tokenizer = AutoTokenizer.from_pretrained(target_tokenizer_path)
        
        # # Decode and print the source and target sequences
        # for i in range(len(src)):
        #     src_text = [self.source_tokenizer.decode(x, skip_special_tokens=False) for x in src[i].tolist()]
        #     tgt_text = [self.target_tokenizer.decode(x, skip_special_tokens=False) for x in tgt[i].tolist()]
        #     # print(f"\nSample {i + 1}:")
        #     # print(f"{len(src_text)} Source: {src_text}")
        #     # print(f"{len(tgt_text)} Target: {tgt_text}")
        src_mask = (src != self.pad_token_id)

        tgt_mask = generate_tgt_mask(tgt, self.pad_token_id)
        
        loss = self.model(src, tgt, mask=src_mask, attn_mask=tgt_mask)
        
        self.log('train_loss', loss, prog_bar=True)
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
        
        # if self.global_step >= 1:
        #     raise ValueError("Stopping at step 1 as requested for debugging")
        
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step. Same approach as training_step but we log val_loss.
        """
        src, tgt = batch['source'], batch['target']
        src_mask = (src != self.pad_token_id)

        # print("src_shape: ", src.shape)
        # print("tgt_shape: ", tgt.shape)
        # print("src_mask_shape: ", src_mask.shape)
        # print("src_mask: ", src_mask)
        


        loss = self.model(src, tgt, mask=src_mask)

        # print("logits_shape: ", logits.shape)

        # Add label smoothing
        # loss = F.cross_entropy(
        #     logits.permute(0, 2, 1),
        #     tgt,
        #     label_smoothing=self.config.model.label_smoothing
        # )
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
            # For SacreBLEU, references should be a list where each item is a list of references
            # For single reference per source, we need [[ref1], [ref2], ...] format
            references_for_sacrebleu = [[ref] for ref in all_references]
            sacrebleu_score = sacrebleu.corpus_bleu(all_generated, references_for_sacrebleu)
            
            # For NLTK BLEU, we need to tokenize and structure references correctly
            tokenized_generated = [gen.split() for gen in all_generated]
            # Each reference needs to be a list in a list - [[tokens1], [tokens2], ...]
            tokenized_references = [[ref.split()] for ref in all_references]
            nltk_bleu = nltk.translate.bleu_score.corpus_bleu(tokenized_references, tokenized_generated) * 100
            
            self.log('val_sacrebleu', sacrebleu_score.score)
            self.log('val_nltk_bleu', nltk_bleu)
            print(f"Validation SacreBLEU: {sacrebleu_score.score:.2f}, NLTK BLEU: {nltk_bleu:.2f}")
        
        self.my_val_outputs.clear()
    
    def sample_translation(self, src, nb_samples=4):
        """Generate translation for a given source batch using the model's generate method."""
        src_mask = (src != self.pad_token_id)
        start_tokens = torch.full((src.shape[0], 1), fill_value=self.config.data.bos_token, dtype=torch.long, device=src.device)
        
        # Use the model's generate method with appropriate parameters
        generated = self.model.generate(
            src,
            mask=src_mask,
            seq_out_start=start_tokens,
            seq_len=self.config.data.target_sequence_length,
            temperature=self.config.get('sample', {}).get('temperature', 1.0)
        )
        
        return generated

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Called by Trainer.predict(...) or Trainer.test(...) with 'predict_step'.
        Generates an output autoregressively.
        """
        src = batch['source']
        src_mask = (src != self.pad_token_id)

        # The x_transformers library offers .generate() for autoregressive decoding
        # 'seq_len' sets the maximum generated length.
        output = self.model.generate(
            src,
            mask=src_mask,
            seq_len=self.config.data.max_tgt_len
        )
        return output

    def configure_optimizers(self):
        """
        Define your optimizer. (Could also add schedulers, etc.)
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.model.peak_lr,
            betas=self.config.optimizer.betas,
            eps=self.config.optimizer.eps,
            weight_decay=self.config.optimizer.weight_decay
        )
        
        # Add the same LR scheduler as Sundae
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