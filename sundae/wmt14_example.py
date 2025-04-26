import datasets
from datasets import load_dataset
import os
from typing import Dict, List, Tuple
from datasets import Dataset
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
    dataset = dataset.shuffle(seed=42)
    
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

def get_toy_wmt14_ende_data(
    splits: list = ["train", "test"],
    size: int = 1000,
    test_size: float = 0.2,
    seed: int = 42,
    max_length: int = None,
    reverse: bool = False,
) -> dict:
    """
    Build a toy EN–DE dataset of given size by re-phrasing 6 base sentences,
    split into train/test, and run through preprocess_wmt14.
    """
    # 1) define the 6 base pairs and a few paraphrases each
    paraphrases = {
        ("Hello.", "Hallo."): [
            ("Hello.",                             "Hallo."),
            ("Hello!",                            "Hallo!"),
            ("Hi.",                               "Hi."),
            ("Hi!",                              "Hi!"),
            ("Hey there.",                       "Hey dort."),
            # longer variants
            ("Hello again, it’s been a while.",    "Hallo nochmal, es ist schon eine Weile her."),
            ("Hello and welcome to our session.",  "Hallo und willkommen zu unserer Sitzung."),
        ],
        ("How are you?", "Wie geht's?"): [
            ("How are you?",                     "Wie geht's?"),
            ("How are you doing?",               "Wie läuft's?"),
            ("How's it going?",                  "Wie geht es?"),
            ("How have you been?",               "Wie bist du gewesen?"),
            # longer variants
            ("How are you today, my friend?",     "Wie geht's dir heute, mein Freund?"),
            ("How are you feeling this morning?", "Wie fühlst du dich heute Morgen?"),
        ],
        ("Good morning.", "Guten Morgen."): [
            ("Good morning.",                     "Guten Morgen."),
            ("Morning!",                          "Morgen!"),
            ("Good day.",                         "Guten Tag."),
            # longer variants
            ("Good morning, I hope you slept well.", "Guten Morgen, ich hoffe, du hast gut geschlafen."),
            ("Good morning to everyone here today!",  "Guten Morgen an alle hier heute!"),
        ],
        ("See you later!", "Bis später!"): [
            ("See you later!",                    "Bis später!"),
            ("See you soon!",                     "Bis bald!"),
            ("Catch you later!",                  "Wir sehen uns später!"),
            # longer variants
            ("See you later at the usual place!",  "Wir sehen uns später am üblichen Ort!"),
            ("See you later, take care until then!", "Bis später, pass bis dahin auf dich auf!"),
        ],
        ("Thank you very much.", "Vielen Dank."): [
            ("Thank you very much.",              "Vielen Dank."),
            ("Thanks a lot.",                     "Danke vielmals."),
            ("Thank you so much.",                "Danke sehr."),
            # longer variants
            ("Thank you very much for your help.",  "Vielen Dank für deine Hilfe."),
            ("Thank you very much, I really appreciate it.", "Vielen Dank, ich weiß es sehr zu schätzen."),
        ],
    }

    # 2) flatten into a list, then sample with replacement up to `size`
    all_pairs = []
    for variants in paraphrases.values():
        all_pairs.extend(variants)

    import random
    random.seed(seed)
    toy_list = [random.choice(all_pairs) for _ in range(size)]
    # wrap as “translation” dicts
    toy_data = [
        {"translation": {"en": en, "de": de}}
        for en, de in toy_list
    ]

    # 3) `Dataset` + shuffle + train_test_split
    ds = Dataset.from_list(toy_data).shuffle(seed=seed)
    split_ds = ds.train_test_split(test_size=test_size, seed=seed)

    # 4) preprocess each split
    data = {}
    for split in splits:
        raw = split_ds[split]
        data[split] = preprocess_wmt14(raw, max_length=max_length, reverse=reverse)

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