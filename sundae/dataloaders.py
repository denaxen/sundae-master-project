from torch.utils.data import DataLoader
from data import get_text8
from wmt14_example import get_wmt14_ende_data
from transformers import AutoTokenizer

def tokenize_wmt_example(example, en_tokenizer, de_tokenizer, add_special_tokens=True, max_length=None):
    # Tokenize the source and target fields.
    # (Assumes that the source field contains the English text and target the German text, by default.)
    example["source"] = en_tokenizer.encode(
        example["source"], 
        add_special_tokens=add_special_tokens,
        padding="max_length" if max_length else None,
        max_length=max_length,
        truncation=True if max_length else None
    )
    example["target"] = de_tokenizer.encode(
        example["target"], 
        add_special_tokens=add_special_tokens,
        padding="max_length" if max_length else None,
        max_length=max_length,
        truncation=True if max_length else None
    )
    return example

def get_dataloaders(config):
    """Return the dataloader for the experiment."""
    
    if config.data.name.lower() == "text8":
        # Get datasets
        train_dataset, eval_dataset = get_text8(
            config.data.root, seq_len=config.data.sequence_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.loader.batch_size,
            num_workers=config.loader.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.data.loader.batch_size,
            num_workers=config.loader.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    elif config.data.name.lower() == "wmt14-ende":
        # Load the WMT14 dataset; here we assume we're doing English-to-German.
        # You can adjust the language pair or use a config flag (e.g., reverse) as needed.
        data = get_wmt14_ende_data(
            splits=["train", "test"],
            max_length=config.data.max_length,  # e.g., maximum token length for filtering
            reverse=config.data.get("reverse", False),
            cache_dir=config.data.root
        )
        # Load tokenizers; assume paths are provided in config.
        en_tokenizer = AutoTokenizer.from_pretrained(config.data.en_tokenizer_path)
        de_tokenizer = AutoTokenizer.from_pretrained(config.data.de_tokenizer_path)
        
        # Apply tokenization to each example in train and validation splits.
        for split in ["train", "test"]:
            # data[split] = data[split].select(range(2000))
            data[split] = data[split].map(
                lambda x: tokenize_wmt_example(
                    x, 
                    en_tokenizer, 
                    de_tokenizer, 
                    max_length=config.data.max_length
                ),
                remove_columns=["translation", "source_lang", "target_lang"],
            )
            # Set format to return PyTorch tensors for the tokenized columns.
            data[split].set_format("torch", columns=["source", "target"])
        
        # Create dataloaders for WMT14. (We use only train and validation for training and evaluation.)
        train_loader = DataLoader(
            data["train"],
            batch_size=config.data.loader.batch_size,
            num_workers=config.loader.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            data["test"],
            batch_size=config.data.loader.batch_size,
            num_workers=config.loader.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    else:
        raise ValueError(f"Unknown dataset type: {config.data.name}")
    
    return train_loader, eval_loader 