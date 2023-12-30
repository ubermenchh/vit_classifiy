import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision, cv2, numpy as np, matplotlib.pyplot as plt, os
from config import Config

if "KAGGLE_URL_BASE" in os.environ:
    train_dir = "/kaggle/input/100-bird-species/train"
    valid_dir = "/kaggle/input/100-bird-species/valid"
    test_dir = "/kaggle/input/100-bird-species/test"
else:
    train_dir = "./data/train/"
    valid_dir = "./data/valid/"
    test_dir = "./data/test/"


def Dataset(bs, crop_size, sample_size="full"):
    transformations = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(crop_size),
            torchvision.transforms.CenterCrop(crop_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    if sample_size == "full":
        train_data = torchvision.datasets.ImageFolder(
            root=train_dir, transform=transformations
        )
        valid_data = torchvision.datasets.ImageFolder(
            root=valid_dir, transform=transformations
        )
        test_data = torchvision.datasets.ImageFolder(
            root=test_dir, transform=transformations
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=bs
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_data, shuffle=True, batch_size=bs
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_data, shuffle=True, batch_size=bs
        )
    else:
        train_data = torchvision.datasets.ImageFolder(
            root=train_dir, transform=transformations
        )
        indices = torch.arange(sample_size)
        train_data = torch.utils.data.Subset(train_data, indices)
        train_dataloader = torch.utils.data.DataLoader(
            train_data, shuffle=True, batch_size=bs
        )

    valid_data = torchvision.datasets.ImageFolder(
        root=valid_dir, transform=transformations
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, shuffle=True, batch_size=bs
    )

    test_data = torchvision.datasets.ImageFolder(
        root=test_dir, transform=transformations
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, shuffle=True, batch_size=bs
    )

    return train_dataloader, valid_dataloader, test_dataloader

