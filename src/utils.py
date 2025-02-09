import torch
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import pickle
from typing import Any


def set_seed(seed: int = 42) -> None:
    """
    Fixes random seed for reproducibility across multiple libraries.

    Parameters:
    seed (int): The seed value to use for all random number generators to ensure reproducibility.
                Defaults to 42.

    Returns:
    None
    """
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    torch.cuda.manual_seed(seed)  # PyTorch, for CUDA
    torch.cuda.manual_seed_all(seed)  # PyTorch, if using multi-GPU
    os.environ["PYTHONHASHSEED"] = str(seed)  # Python hash seed


def r2_score(y_true: Tensor, y_pred: Tensor) -> Tensor:
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def to_class_labels(y, min_val=-5, max_val=5, increment=0.25):
    # Convert continuous values to class labels
    class_labels = torch.round((y - min_val) / increment).long()

    return class_labels


def from_class_labels(class_labels, min_val=-5, max_val=5, increment=0.25):
    # Convert class labels back to continuous values
    values = (class_labels.float() * increment) + min_val
    return values


def sample_from_distribution(prob_dist, min_val=-5, max_val=5, increment=0.25):
    """
    Samples from the given probability distribution and converts the samples to continuous values.

    Parameters:
    - prob_dist (torch.Tensor): The probability distribution from which to sample, shape (batch, 40).
    - min_val (float): The minimum value represented by the classes.
    - max_val (float): The maximum value represented by the classes.
    - increment (float): The increment between class values.

    Returns:
    - torch.Tensor: Sampled continuous values, shape (batch, 1).
    """
    # Ensure the probability distribution sums to 1 along the last dimension
    prob_dist = prob_dist / prob_dist.sum(dim=1, keepdim=True)

    # Sample from the probability distribution
    # torch.multinomial requires the probabilities to be on CPU
    class_labels = torch.multinomial(prob_dist.cpu(), num_samples=1).to(
        prob_dist.device
    )

    # Convert sampled class labels to continuous values
    values = from_class_labels(class_labels, min_val, max_val, increment)

    return values


def plot_predictions(
    dataloader,
    model,
    max_len=500,
    title="Comparison of Actual and Predicted Data",
    categorical_prob=False,
):
    model.eval()  # Set the model to evaluation mode
    model.to("cpu")  # Ensure the model is on the CPU
    actuals = []
    predictions = []

    with torch.no_grad():  # No need to track gradients
        for inputs, targets in dataloader:
            inputs, targets = inputs.to("cpu"), targets.to(
                "cpu"
            )  # Ensure data and model are on the same device
            if categorical_prob:
                targets = to_class_labels(targets)
            outputs = model(inputs)
            if categorical_prob:
                outputs = F.softmax(outputs, dim=1)
                outputs = sample_from_distribution(outputs)
                targets = from_class_labels(targets)

            # outputs = torch.round(outputs*4)/4  # Round the predictions to the nearest integer
            # Convert predictions and actuals to lists for plotting
            predictions.extend(outputs.view(-1).tolist())
            actuals.extend(targets.tolist())

            if (
                len(actuals) >= max_len
            ):  # Limit the number of points to plot for clarity
                break

    actuals = actuals[:max_len]
    predictions = predictions[:max_len]

    # Generating the plot
    plt.figure(figsize=(10, 6))
    plt.ylim(-1, 1)
    plt.plot(actuals, label="Actual Data", color="blue", alpha=0.6)
    plt.plot(predictions, label="Predicted Data", color="red", alpha=0.6)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def save_dataclass(instance: Any, file_path: str) -> None:
    """
    Saves an instance of a dataclass to a file using pickle.

    Parameters:
    - instance (Any): The dataclass instance to save.
    - file_path (str): The path to the file where the dataclass instance will be saved.

    Returns:
    - None
    """
    with open(file_path, "wb") as file:
        pickle.dump(instance, file)


def load_dataclass(file_path: str) -> Any:
    """
    Loads an instance of a dataclass from a file using pickle.

    Parameters:
    - file_path (str): The path to the file from which the dataclass instance will be loaded.

    Returns:
    - Any: The loaded dataclass instance.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)


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
