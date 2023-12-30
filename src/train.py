import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import neptune
from tqdm import tqdm

from transformer import VisionTransformer


def neptune_monitoring(config):
    PARAMS = {}
    for key, val in config.__dict__.items():
        if key not in ["__module__", "__dict__", "__weakref__", "__doc__"]:
            PARAMS[key] = val
    return PARAMS


def train_engine(
    n_epochs, train_data, valid_data, model, optimizer, loss_fn, device, monitoring=True
):
    train_accuracy = 0
    valid_accuracy = 0
    best_accuracy = 0

    for epoch in range(1, n_epochs + 1):
        total = 0
        with tqdm(train_data, unit="iteration") as train_epoch:
            train_epoch.set_description((f"Epoch {epoch}"))
            for i, (data, target) in enumerate(train_epoch):
                total_samples = len(train_data.dataset)

                model = model.to(device)
                x = data.to(device)
                y = target.to(device)
                optimizer.zero_grad()

                logits, attn_weights = model(x)
                probs = F.log_softmax(logits, dim=1)
                loss = F.nll_loss(probs, y, reduction="sum")
                loss.backward()
                optimizer.step()

                _, pred = torch.max(logits, dim=1)
                train_accuracy += torch.sum(pred == y).item()
                total += target.size(0)
                accuracy_ = 100 * train_accuracy / total
                train_epoch.set_postfix(loss=loss.item(), accuracy=accuracy_)

                if monitoring:
                    run["Training_loss"].log(loss.item())
                    run["Training_acc"].log(accuracy_)

                if accuracy_ > best_accuracy:
                    best_accuracy = accuracy_
                    best_model = model
                    torch.save(best_model, f"../metadata/model.pth")

        total_samples = len(valid_data.dataset)
        correct_samples = 0
        total_ = 0
        model.eval()
        with torch.inference_mode():
            with tqdm(valid_data, unit="iteration") as valid_epoch:
                valid_epoch.set_description(f"Epoch {epoch}")
                for i, (data, target) in enumerate(valid_epoch):
                    model = model.to(device)
                    x = data.to(device)
                    y = target.to(device)

                    logits, attn_weights = model(x)
                    probs = F.log_softmax(logits, dim=1)
                    valid_loss = F.nll_loss(probs, y, reduction="sum")

                    _, pred = torch.max(logits, dim=1)
                    valid_accuracy += torch.sum(pred == y).item()
                    total_ += target.size(0)
                    valid_accuracy_ = 100 * valid_accuracy / total_
                    valid_epoch.set_postfix(
                        loss=valid_loss.item(), accuracy=valid_accuracy_
                    )

                    if monitoring:
                        run["Valid_accuracy "].log(valid_accuracy_)
                        run["Valid_loss"].log(loss.item())


if __name__ == "__main__":
    from preprocessing import Dataset
    from config import Config

    dataset_name = "gpiosenka/100-bird-species"
    if "KAGGLE_URL_BASE" not in os.environ:
        import kaggle
        api = kaggle.api 
        api.authenticate()
        api.dataset_download_files(dataset_name, path="./data/", unzip=True)

    config = Config()
    params = neptune_monitoring(Config)
    run = neptune.init_run(
        project="ubermenchh/ViT-bird-classification",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNGJjZDVkNy1iZWFhLTRhNzEtYWI4Ni1lODVlZjIzODM2MDMifQ==",
    )
    run["parameters"] = params

    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        linear_dim=config.LINEAR_DIM,
        dropout_rate=config.DROPOUT_RATE,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
        eps=config.EPS,
        std_norm=config.STD_NORM,
    )

    train_data, valid_data, test_data = Dataset(
        config.BATCH_SIZE, config.IMG_SIZE, config.DATASET_SAMPLE
    )

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_engine(
        n_epochs=config.N_EPOCHS,
        train_data=train_data,
        valid_data=valid_data,
        model=model,
        optimizer=optimizer,
        loss_fn="nll_loss",
        device=config.DEVICE[1],
        monitoring=True,
    )
