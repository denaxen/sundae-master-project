from dataloaders import get_dataloaders
import hydra
from tqdm import tqdm
from transformers import AutoTokenizer
from wmt14_example import load_wmt14_ende

@hydra.main(version_base=None, config_path="configs", config_name="ar_mt_hf_transformer")
def main(config):
    train_loader, eval_loader = get_dataloaders(config)
    tokenizer = AutoTokenizer.from_pretrained("speedcell4/wmt14-deen-shared-40k")
    
    batch_idx = 0
    for i, batch in enumerate(tqdm(train_loader)):
        # print(batch)
        if i == 47420:
        # if i == 10:
            source, target = batch['source'], batch['target']
            for s, t in zip(source, target):
                s_string = tokenizer.decode(s, skip_special_tokens=True)
                t_string = tokenizer.decode(t, skip_special_tokens=True)
                print(f"Source: {s_string}")
                print(f"Target: {t_string}")
                print("-"*100) 
            break

    
    # Print a few examples from the evaluation dataset
    # print("\nEvaluation dataset examples:")
    # for batch in eval_loader:
    #     print(batch)
    #     print(batch.shape)
    #     break

if __name__ == "__main__":
    # main()
    dataset = load_wmt14_ende(split="train", cache_dir="data/wmt14")
    dataset = dataset.shuffle(seed=42)
    for i in range(3):
        print(dataset[i])
        # print(f"Source: {dataset[i]['source']}")
        # print(f"Target: {dataset[i]['target']}")
        print("-"*100)