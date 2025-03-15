import os
from pathlib import Path

import hydra
import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from omegaconf import OmegaConf

from dataloaders import get_dataloaders
from loading_utils import get_module
from utils.other_utils import (
    add_resolvers,
    # configure_optimizer,
    fsspec_exists,
    prepare_logger,
)


def train(config):
    logger.info("Starting training")

    if config.get("wandb", None):
        # Wandb args need to be in a dict, not omegaconf object
        wandb_args_dict = OmegaConf.to_object(config.wandb)

        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=OmegaConf.to_object(config),
            **wandb_args_dict,
        )
    else:
        wandb_logger = None
    # Always log in tensorboard
    tb_logger = TensorBoardLogger("tb_logs", name="logs")
    loggers = tb_logger if wandb_logger is None else (wandb_logger, tb_logger)

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and fsspec_exists(config.checkpointing.resume_ckpt_path)
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
        logger.info(f"Training starting from checkpoint at {ckpt_path}")
    else:
        ckpt_path = None
        logger.info("Training starting from scratch (no checkpoint to reload)")

    # Prepare data
    train_loader, eval_loader = get_dataloaders(config)
    lightning_module = get_module(config)

    if config.compile:  # speedup model
        lightning_module.model = torch.compile(lightning_module.model)

    # Create lightning trainer from fields in the config
    trainer = hydra.utils.instantiate(
        config.trainer, default_root_dir=os.getcwd(), strategy="ddp", logger=loggers
    )

    trainer.fit(
        lightning_module,
        train_loader,
        eval_loader,
        ckpt_path=ckpt_path,
    )

    # Save the model if configured to do so
    if isinstance(config.save_model, str):
        if config.save_model.lower() == "false":
            config.save_model = False
        elif config.save_model.lower() == "true":
            config.save_model = True
        else:
            config.save_model = False
    if config.save_model:
        save_path = Path(config.save_model_path)

        # Ensure the parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving model to {save_path}")

        # Save the model's state_dict
        torch.save(lightning_module.model.state_dict(), save_path)
        logger.info("Model saved successfully.")

    validation_results = trainer.validate(
        lightning_module,
        dataloaders=eval_loader,
        verbose=True,  # Optional: Set to True to print validation results
    )

    # Optionally, log or process the validation results
    logger.info(f"Final validation results: {validation_results}")


def eval(config):
    # Run the evaluation
    raise NotImplementedError


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    if hasattr(config, "seed"):
        L.seed_everything(config.seed)
    else:
        L.seed_everything(0)
    # if config.config_optimizer_and_batch_automatically:
    #     configure_optimizer(config)
    if config.loader.global_batch_size < config.loader.batch_size:
        config.loader.batch_size = config.loader.global_batch_size

    # OmegaConf.save(config=config, f=Path(os.getcwd()) / "config.yaml")

    logger.info(f"Arguments:\n{OmegaConf.to_yaml(config, resolve=True)}")
    mode = config.mode

    if mode == "train":
        logger.add(Path(os.getcwd()) / "logs_train.txt")
        train(config)
    elif mode == "eval":
        logger.add(Path(os.getcwd()) / "logs_eval.txt")
        eval(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    add_resolvers()
    prepare_logger()
    main() 