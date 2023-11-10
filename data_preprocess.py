
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torchvision
import torch


def split_dataset(dataset:torchvision.datasets, split_size:float=0.2, seed:int=1126):

    split1, split2 = torch.utils.data.random_split(dataset = dataset,
                                                   lengths=[split_size, 1-split_size],
                                                   generator=torch.Generator().manual_seed(seed))
    # Print out info
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {len(split1)} ({int(split_size*100)}%), {len(split2)} ({int((1-split_size)*100)}%)")

    return split1, split2


def create_dataloaders(batch_size: int = 32,
                       split_size: float = 0.2,
                       train_transform: transforms.Compose = None,
                       test_transform: transforms.Compose = None):

    data_dir = Path("data")

    # Get training data / testing data
    train_data = datasets.Food101(root=data_dir,
                                  split="train", 
                                  transform=train_transform,
                                  download=True) 

    test_data = datasets.Food101(root=data_dir,
                                 split="test",
                                 transform=test_transform,
                                 download=True)

    # Get class names
    class_names = train_data.classes

    ## sample data
    train_data, _ = split_dataset(dataset=train_data,
                                  split_size=0.2)

    test_data, _ = split_dataset(dataset=test_data,
                                 split_size=0.2)

    # Turn images into data loaders
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size,
                                 shuffle=False)

    return train_dataloader, test_dataloader, class_names