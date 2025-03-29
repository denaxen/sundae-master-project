from dataloaders import get_dataloaders
import hydra

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    train_loader, eval_loader = get_dataloaders(config)
    
    # Print a few examples from the training dataset
    print("Training dataset examples:")
    for batch in train_loader:
        print(batch)
        print(batch.shape)
        break
    
    # Print a few examples from the evaluation dataset
    print("\nEvaluation dataset examples:")
    for batch in eval_loader:
        print(batch)
        print(batch.shape)
        break

if __name__ == "__main__":
    # main()
    
