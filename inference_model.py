# Standard library imports
import time
import argparse
from datetime import datetime
from dataclasses import dataclass, fields

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# Local application/library specific imports
from src.transformer_utils import TransformerWithTimeEmbeddings
from src.utils import r2_score, load_dataclass, Config
from src.dataloader import (
    train_test_split,
    OrderBookDataset,
    load_stats_and_normalize,
    apply_fft_high_pass_filter,
    generate_orderbook_features,
    apply_pca_keep_variance,
)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inference script to calculate R2 score from a CSV dataset containing Orderbook ."
    )
    parser.add_argument(
        "csv_path", type=str, help="Path to the CSV file containing the dataset."
    )
    args = parser.parse_args()
    print(f"Loading dataset ... ")
    # Load the dataset
    data_path = args.csv_path
    data = pd.read_csv(data_path)

    # Ensure the target variable is named 'y'
    calculate_R2 = True
    if "y" not in data.columns:
        data["y"] = 0
        calculate_R2 = False

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set the training configuration
    config = load_dataclass("final_model/config.pkl")

    print(f"Preparing dataset ...")
    # Reduce the number of features
    if config.reduce_features:
        data = generate_orderbook_features(data)
    if config.reduce_features_svd:
        data = apply_pca_keep_variance(
            data, variance_threshold=config.svd_variance_threshold
        )

    # Remove very low frequency components to make model more generalizable to unseen data
    data = apply_fft_high_pass_filter(data, config.fft_cutoff)

    # Load the feature statistics and normalize the data
    data = load_stats_and_normalize(data, "final_model/feature_stats.csv")

    # Create dataloader
    dataset = OrderBookDataset(data, config.window_size)
    test_dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    input_size = data.shape[1] - 1  # Number of features, excluding target variable

    transformer_kwargs = {
        "n_out": 1,
        "emb": config.hidden_size,
        "heads": config.transformer_heads,
        "depth": config.transformer_depth,
        "dropout": config.dropout_rate,
    }
    if config.use_transformer:
        model = TransformerWithTimeEmbeddings(
            input_size=input_size, nband=1, **transformer_kwargs
        ).to(device)

    # Loading the model
    model.load_state_dict(
        torch.load("final_model/model.pt", map_location=torch.device(device))
    )
    model.eval()
    print("Calculating R2 ... ")
    y_true_test, y_pred_test = [], []
    for inputs, targets in tqdm(test_dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        y_true_test.append(targets)
        y_pred_test.append(outputs.squeeze().detach())

    y_true_test = torch.cat(y_true_test)
    y_pred_test = torch.cat(y_pred_test)
    df = pd.DataFrame(y_pred_test.cpu().numpy())

    # Save CSV file
    df.to_csv("prediction.csv", index=False)

    if calculate_R2:
        test_r2 = r2_score(y_true_test, y_pred_test)
        print(f"Test R2 score: {test_r2.item()}")
    else:
        print(
            f"No 'y' column found in the dataset. Prediction saved to 'prediction.csv'."
        )
