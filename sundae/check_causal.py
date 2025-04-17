import logging
import os
import torch
import hydra
import sys
from omegaconf import OmegaConf

from loading_utils import get_module

# -------------------------------------------------------------------------
# Hydra entrypoint to load config and test patches
# -------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("check_causal")

    logger.info("Instantiating model (with patched attention mask)...")
    lightning_module = get_module(config)
    model = lightning_module.model
    model.eval()

    batch_size = min(2, getattr(config.loader, 'global_batch_size', 2))
    seq_len    = config.data.target_sequence_length
    vocab_size = config.data.vocabulary_size
    hid_dim    = config.model.embedding_dim

    logger.info(f"Creating dummy input_ids of shape ({batch_size}, {seq_len})")
    input_ids      = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    # Dummy encoder outputs for cross‐attention
    encoder_hidden_states = torch.randn(batch_size, seq_len, hid_dim)
    encoder_attention_mask = torch.ones_like(input_ids)
    # ─── Run Decoder & Grab Attentions ───────────────────────────────────────────
    logger.info("Running decoder with output_attentions=True…")
    with torch.no_grad():
        dec_outputs = model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=True,
        )
    attentions = dec_outputs.attentions  # tuple: one Tensor (B, H, S, S) per layer
    # ─── Check for Bidirectionality ──────────────────────────────────────────────
    # Build a mask that picks out positions j > i (upper triangle, excluding diag)
    future_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    logger.info("Inspecting each decoder self-attention layer for non-zero future attention…")
    for idx, layer_attn in enumerate(attentions):
        # layer_attn: (batch, heads, seq_len, seq_len)
        # any positive weight in j>i slots?
        if (layer_attn[..., future_mask] > 0).any().item():
            logger.info(f"✔ Layer {idx}: found non-zero future attention → bidirectional")
        else:
            logger.error(f"✘ Layer {idx}: all future slots zero → still causal!")
            sys.exit(1)

    logger.info("✅ All decoder self-attention layers are fully bidirectional.")


if __name__ == "__main__":
    main()
