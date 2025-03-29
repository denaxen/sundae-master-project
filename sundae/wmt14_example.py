import datasets
from datasets import load_dataset
import os
from typing import Dict, List, Tuple

def load_wmt14_ende(split: str = "train", cache_dir: str = None) -> datasets.Dataset:
    """
    Load the WMT14 English-German translation dataset for a specific split.
    
    Args:
        split: The dataset split to load ("train", "validation", or "test")
        cache_dir: Optional directory to cache the downloaded dataset
        
    Returns:
        A datasets.Dataset object containing the requested split
    """
    # WMT14 is available in the datasets library
    dataset = load_dataset("wmt14", "de-en", split=split, cache_dir=cache_dir)
    
    # The dataset has 'translation' column with 'en' and 'de' subfields
    return dataset

def preprocess_wmt14(dataset: datasets.Dataset, 
                     max_length: int = None, 
                     reverse: bool = False) -> datasets.Dataset:
    """
    Preprocess the WMT14 dataset for training.
    
    Args:
        dataset: The dataset to preprocess
        max_length: Optional maximum sequence length (will filter out longer sequences)
        reverse: If True, swap source and target (de->en instead of en->de)
        
    Returns:
        Preprocessed dataset with 'source' and 'target' columns
    """
    def process_translation(example):
        if reverse:
            # German to English
            source, target = example["translation"]["de"], example["translation"]["en"]
        else:
            # English to German (default)
            source, target = example["translation"]["en"], example["translation"]["de"]
        
        return {
            "source": source,
            "target": target,
            "source_lang": "de" if reverse else "en",
            "target_lang": "en" if reverse else "de"
        }
    # dataset = dataset.select(range(3000))
    processed = dataset.map(process_translation)
    
    # Filter by length if max_length is specified
    if max_length:
        processed = processed.filter(lambda x: len(x["source"].split()) <= max_length and 
                                             len(x["target"].split()) <= max_length)
    
    return processed

def get_wmt14_ende_data(splits: List[str] = ["train", "validation", "test"], 
                        max_length: int = None,
                        reverse: bool = False,
                        cache_dir: str = None) -> Dict[str, datasets.Dataset]:
    """
    Load and preprocess the WMT14 English-German dataset.
    
    Args:
        splits: List of splits to load (train, validation, test)
        max_length: Optional maximum sequence length
        reverse: If True, swap source and target (de->en instead of en->de)
        cache_dir: Optional directory to cache the downloaded dataset
        
    Returns:
        Dictionary mapping split names to preprocessed datasets
    """
    data = {}
    
    for split in splits:
        # Load the raw dataset
        raw_dataset = load_wmt14_ende(split=split, cache_dir=cache_dir)
        
        # Preprocess the dataset
        processed_dataset = preprocess_wmt14(raw_dataset, max_length=max_length, reverse=reverse)
        
        data[split] = processed_dataset
    
    return data

if __name__ == "__main__":
    from transformers import AutoTokenizer

    shared_tokenizer = AutoTokenizer.from_pretrained("tokenizers/wmt14-deen-shared-40k")
    print("Vocab size:", shared_tokenizer.vocab_size)
    print("Special tokens:", shared_tokenizer.all_special_tokens)
    print("Special tokens mapping:", shared_tokenizer.special_tokens_map)

    data = get_wmt14_ende_data(["train", "validation", "test"], max_length=128, cache_dir="data/wmt14")
    print(data)
    for split in data:
        print(f"Split: {split}")
        print(f"Size: {len(data[split])}")
        print(f"Sample: {data[split][0]}")
        print()
    validation_data = data["validation"]

    for i in range(3):
        print(f"Source ({validation_data[i]['source_lang']}): {validation_data[i]['source']}")
        print(f"Target ({validation_data[i]['target_lang']}): {validation_data[i]['target']}")
        print()

    print(shared_tokenizer.encode(validation_data[0]["source"]))
    print(shared_tokenizer.encode(validation_data[0]["target"]))