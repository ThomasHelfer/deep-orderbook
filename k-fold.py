# Standard library imports
import time
import os
from datetime import datetime
from dataclasses import dataclass, fields

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from sklearn.model_selection import TimeSeriesSplit

# Local application/library specific imports
from src.transformer_utils import TransformerWithTimeEmbeddings
from src.utils import (
    r2_score,
    plot_predictions,
    set_seed,
    save_dataclass,
    from_class_labels,
    to_class_labels,
    sample_from_distribution,
)
from src.dataloader import *
from src.LSTM_utils import DeepLSTM


@dataclass
class Config:
    window_size: int = 100
    train_size: float = 0.5
    batch_size: int = 2**13

    categorical_prob: bool = False  # Makes a class prediction instead of an regression
    use_transformer: bool = True
    transformer_depth: int = 1
    transformer_heads: int = 2
    LSTM_depth: int = 6
    dropout_rate: float = 0.0001
    weight_decay: float = 0.01

    reduce_features: bool = True
    reduce_features_svd: bool = False
    svd_variance_threshold: float = 0.8

    hidden_size: int = 32
    epochs: int = 5
    learning_rate: int = 0.0005

    seed: int = 1

    fft_cutoff: float = 0.0005

    differentiate: bool = False

    write_out_dir = "final_model"


# Set the training configuration
config = Config()

set_seed(config.seed)

data_path = "../data.csv"
data = pd.read_csv(data_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_workers = 12


# Check and create dir
if not os.path.exists(config.write_out_dir):
    os.makedirs(config.write_out_dir)

if config.reduce_features:
    data = generate_orderbook_features(data)
if config.reduce_features_svd:
    data = apply_pca_keep_variance(
        data, variance_threshold=config.svd_variance_threshold
    )

tscv = TimeSeriesSplit(n_splits=5)

counterfold = 1


for train_index, test_index in tscv.split(data):
    print(f"this is fold number {counterfold}")
    counterfold += 1
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

    train_data = apply_fft_high_pass_filter(train_data, config.fft_cutoff)
    test_data = apply_fft_high_pass_filter(test_data, config.fft_cutoff)
    # Normalizing data
    if config.reduce_features or config.reduce_features_svd:
        train_data, test_data = normalize_train_test(
            train_data, test_data, config.write_out_dir + "/feature_stats.csv"
        )
    else:
        # If using the full feature set, we normalize all features that a rates or sizes together, as they are of the same type
        Rate_features = [col for col in train_data.columns if "Rate" in col]
        Size_features = [col for col in train_data.columns if "Size" in col]

        apply_group_normalization(train_data, test_data, [Rate_features, Size_features])

    # Creating Dataset and DataLoader for both training and testing
    train_dataset = OrderBookDataset(train_data, config.window_size)
    test_dataset = OrderBookDataset(test_data, config.window_size)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    input_size = data.shape[1] - 1  # Number of features, excluding target variable
    output_size = 40 if config.categorical_prob else 1  # Predicting a single value

    transformer_kwargs = {
        "n_out": output_size,
        "emb": config.hidden_size,
        "heads": config.transformer_heads,
        "depth": config.transformer_depth,
        "dropout": config.dropout_rate,
    }
    if config.use_transformer:
        model = TransformerWithTimeEmbeddings(
            input_size=input_size, nband=1, **transformer_kwargs
        ).to(device)
    else:
        model = DeepLSTM(
            input_size,
            config.hidden_size,
            num_layers=config.LSTM_depth,
            output_size=output_size,
            dropout_rate=config.dropout_rate,
        )

    loss_function = nn.MSELoss()  # Mean Squared Error Loss for regression tasks
    cross_loss_function = (
        nn.CrossEntropyLoss()
    )  # Cross Entropy Loss for treating the task as a classification problem

    # Getting current time for unique id
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    tensorboard_path = f"/content/drive/MyDrive/EXYCISE/"
    # If the path does not exist, write in local directory
    if not os.path.exists(tensorboard_path):
        tensorboard_path = "./"

    writer = SummaryWriter(tensorboard_path + "logs_new/experiment_{current_time}")

    print(
        f"Total number of parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    model.to(device)
    track_R2_train = []
    track_R2_test = []
    for epoch in range(config.epochs):
        model.train()

        train_losses = []
        y_true_train, y_pred_train = [], []

        # Setting up Progress bar
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if config.categorical_prob:
                targets = to_class_labels(targets)

            outputs = model(inputs)

            if config.categorical_prob:
                loss = cross_loss_function(outputs, targets)
                targets = from_class_labels(targets.detach())
                outputs = sample_from_distribution(outputs.detach())

            else:
                loss = loss_function(outputs, targets.unsqueeze(1))

            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
            }
            progress_bar.set_postfix(**logs)

            y_true_train.append(targets)
            y_pred_train.append(outputs.squeeze().detach())

        progress_bar.close()

        y_true_train = torch.cat(y_true_train)
        y_pred_train = torch.cat(y_pred_train)

        train_r2 = r2_score(y_true_train, y_pred_train)

        model.eval()
        test_losses = []
        y_true_test, y_pred_test = [], []
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                if config.categorical_prob:
                    targets = to_class_labels(targets)

                outputs = model(inputs)

                if config.categorical_prob:
                    loss = cross_loss_function(outputs, targets)
                    targets = from_class_labels(targets.detach())
                    outputs = sample_from_distribution(outputs.detach())
                else:
                    loss = loss_function(outputs, targets.unsqueeze(1))

                test_losses.append(loss.item())

                y_true_test.append(targets)
                y_pred_test.append(outputs.squeeze().detach())

        y_true_test = torch.cat(y_true_test)
        y_pred_test = torch.cat(y_pred_test)
        test_r2 = r2_score(y_true_test, y_pred_test)

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {sum(train_losses)/len(train_losses):.8f} | "
            f"Test Loss: {sum(test_losses)/len(test_losses):.8f} | "
            f"Train R2: {train_r2:.8f} | "
            f"Test R2: {test_r2:.8f}"
        )
        writer.add_scalar(
            "Train Loss", sum(train_losses) / len(train_losses), global_step=epoch
        )
        writer.add_scalar(
            "Test Loss", sum(test_losses) / len(test_losses), global_step=epoch
        )
        writer.add_scalar("Train R2", train_r2, global_step=epoch)
        writer.add_scalar("Test R2", test_r2, global_step=epoch)
        track_R2_train.append(train_r2.to("cpu").detach().numpy())
        track_R2_test.append(test_r2.to("cpu").detach().numpy())
    # Writing hparams and ffinal metrics
    config_dict = config.__dict__
    metric_dict = {
        "Train max R2": max(track_R2_train),
        "Test max R2": max(track_R2_test),
    }
    writer.add_hparams(hparam_dict=config_dict, metric_dict=metric_dict)
    writer.flush()
    writer.close()

# save the model into the local working directory
torch.save(model.state_dict(), config.write_out_dir + "/model.pt")
# save the used config using pickle
save_dataclass(config, config.write_out_dir + "/config.pkl")
