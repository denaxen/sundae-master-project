from torch.utils.data import DataLoader
from data import get_text8

def get_dataloaders(config):
    """Return the dataloader for the experiment."""
    
    # Get datasets
    train_dataset, eval_dataset = get_text8(
        config.data.root, seq_len=config.data.sequence_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        shuffle=True,
        pin_memory=True,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.loader.batch_size,
        num_workers=config.loader.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    
    return train_loader, eval_loader 