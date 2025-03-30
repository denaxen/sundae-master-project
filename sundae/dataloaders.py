from torch.utils.data import DataLoader
from data import get_text8
from wmt14_example import get_wmt14_ende_data
from transformers import AutoTokenizer
import hashlib
import os
import datasets

def tokenize_wmt_example(example, shared_tokenizer, add_special_tokens=True, max_length=None):
    # Tokenize the source and target fields.
    # (Assumes that the source field contains the English text and target the German text, by default.)
    example["source"] = shared_tokenizer.encode(
        example["source"], 
        add_special_tokens=add_special_tokens,
        padding="max_length" if max_length else None,
        max_length=max_length,
        truncation=True if max_length else None
    )
    example["target"] = shared_tokenizer.encode(
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
            drop_last=True,
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config.data.loader.batch_size,
            num_workers=config.loader.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    elif config.data.name.lower() == "wmt14-ende":
        # Create a unique cache key based on dataset parameters
        cache_params = f"maxlen_{config.data.max_length}_rev_{config.data.get('reverse', False)}"
        cache_hash = hashlib.md5(cache_params.encode()).hexdigest()[:10]
        cache_dir = os.path.join(config.data.root, "tokenized_cache")
        train_cache_path = os.path.join(cache_dir, f"train_{cache_hash}")
        test_cache_path = os.path.join(cache_dir, f"test_{cache_hash}")
        
        # Check if we have cached datasets
        train_cached = os.path.exists(train_cache_path)
        test_cached = os.path.exists(test_cache_path)
        
        if train_cached and test_cached:
            print(f"Loading tokenized datasets from cache: {cache_dir}")
            # Load datasets from cache
            data = {
                "train": datasets.load_from_disk(train_cache_path),
                "test": datasets.load_from_disk(test_cache_path)
            }
        else:
            print("Tokenized datasets not found in cache, processing data...")
            # Load the WMT14 dataset; here we assume we're doing English-to-German.
            data = get_wmt14_ende_data(
                splits=["train", "test"],
                max_length=config.data.max_length,
                reverse=config.data.get("reverse", False),
                cache_dir=config.data.root
            )
            shared_tokenizer = AutoTokenizer.from_pretrained(config.data.shared_tokenizer_path)
            
            # Apply tokenization to each example in train and validation splits.
            for split in ["train", "test"]:
                # if split == "train":
                #     data[split] = data[split].select(range(30000))
                data[split] = data[split].map(
                    lambda x: tokenize_wmt_example(
                        x, 
                        shared_tokenizer, 
                        max_length=config.data.max_length
                    ),
                    remove_columns=["translation", "source_lang", "target_lang"],
                )
                # Set format to return PyTorch tensors for the tokenized columns.
                data[split].set_format("torch", columns=["source", "target"])
                
                # Save processed datasets to cache
                cache_path = train_cache_path if split == "train" else test_cache_path
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                data[split].save_to_disk(cache_path)
                print(f"Saved {split} dataset to {cache_path}")
        
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