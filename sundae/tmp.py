from dataloaders import get_dataloaders
import hydra
from tqdm import tqdm

@hydra.main(version_base=None, config_path="configs", config_name="ar_mt_hf_transformer")
def main(config):
    train_loader, eval_loader = get_dataloaders(config)
    
    batch_idx = 0
    for batch in tqdm(train_loader):
        # print(batch)
        batch_idx += 1
    print(f"Batch index: {batch_idx}")

    
    # Print a few examples from the evaluation dataset
    # print("\nEvaluation dataset examples:")
    # for batch in eval_loader:
    #     print(batch)
    #     print(batch.shape)
    #     break

if __name__ == "__main__":
    main()