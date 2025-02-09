from typing import Tuple
from typing import List

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class OrderBookDataset(Dataset):
    """
    A PyTorch Dataset for order book data.

    Attributes:
        data (pd.DataFrame): The dataset containing the order book data.
        window_size (int): The size of the window to use for each sample, i.e., the number of time steps.

    Methods:
        __len__: Returns the total number of samples available in the dataset.
        __getitem__: Returns a single sample from the dataset, including input features and target.
    """

    def __init__(self, data: pd.DataFrame, window_size: int):
        """
        Initializes the dataset with the provided data and window size.

        Parameters:
            data (pd.DataFrame): The dataset to load, expected to be a pandas DataFrame.
            window_size (int): The size of the window for each sample.
        """
        self.data = torch.tensor(
            data.fillna(0).values, dtype=torch.float
        )  # Assuming you want to fill NaN values with 0
        self.window_size = window_size

    def __len__(self) -> int:
        """Returns the total number of samples available in the dataset."""
        return len(self.data) - self.window_size - 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Parameters:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the input features and target value
            for the given index, both as PyTorch tensors.
        """
        X = self.data[idx : idx + self.window_size, :-1]
        y = self.data[idx + self.window_size - 1, -1]
        return X, y


class OrderBookDatasetWithFixedStats(Dataset):
    """
    A PyTorch Dataset for order book data, with approximate precomputation of normalization
    statistics based on overlapping chunks.
    """

    def __init__(self, data: pd.DataFrame, window_size: int, chunk_size: int = 10000):
        """
        Initializes the dataset with data, window size, and chunk size for approximate precomputation.

        Parameters:
            data (pd.DataFrame): The dataset.
            window_size (int): The size of the window for output samples.
            chunk_size (int): The size of chunks for precomputation.
        """
        self.data = data.fillna(0)
        self.window_size = window_size
        self.chunk_size = chunk_size

        # Precompute means and stds for chunks
        self.chunk_stats = self._precompute_chunk_stats(data, chunk_size)

    def _precompute_chunk_stats(self, data, chunk_size):
        """
        Precomputes mean and std for overlapping chunks of the dataset.

        Returns:
            A list of tuples containing (mean, std) for each chunk.
        """
        chunk_stats = []
        for start in range(0, len(data), chunk_size // 2):
            end = min(start + chunk_size, len(data))
            chunk = data.iloc[start:end, :-1]
            chunk_mean = chunk.mean()
            chunk_std = chunk.std().replace(0, 1)  # Avoid division by zero
            chunk_stats.append((chunk_mean, chunk_std))
        return chunk_stats

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx: int):
        # Find the appropriate chunk based on index
        chunk_idx = idx // (self.chunk_size // 2)
        mean, std = self.chunk_stats[min(chunk_idx, len(self.chunk_stats) - 1)]

        X = self.data.iloc[idx : idx + self.window_size, :-1]
        X_normalized = (X - mean) / std

        y = self.data.iloc[idx + self.window_size - 1, -1]

        return torch.tensor(X_normalized.values, dtype=torch.float), torch.tensor(
            y, dtype=torch.float
        )


def train_test_split(
    data: pd.DataFrame, train_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and testing datasets based on the provided train_size fraction.

    Parameters:
        data (pd.DataFrame): The complete dataset to be split.
        train_size (float): The fraction of the dataset to be used for training (0 < train_size < 1).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing datasets.
    """
    train_end = int(len(data) * train_size)
    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]  # No overlap needed here based on the correction
    return train_data, test_data


def apply_fft_high_pass_filter(df, cutoff=0.0001, fs=1):
    """
    Applies FFT-based high-pass filtering to each column of the DataFrame except the last.

    Parameters:
    - df: pandas DataFrame containing your data.
    - cutoff: Cutoff frequency for the high-pass filter.
    - fs: Sampling rate of your time series data.

    Returns:
    - A new DataFrame with the high-pass filtered signals for each feature column and the original target column.
    """
    # Create a new DataFrame to store the filtered data
    filtered_df = pd.DataFrame()

    # Process each column except the last one
    for column in df.columns[:-1]:  # Exclude the last column
        time_series = df[column].values

        # Perform FFT
        fft_result = np.fft.fft(time_series)
        frequencies = np.fft.fftfreq(len(fft_result), d=1 / fs)

        # Zero out frequencies that are too low (below the cutoff)
        cutoff_index = np.abs(frequencies) < cutoff
        fft_result[cutoff_index] = 0

        # Convert back to time domain
        high_passed_signal = np.fft.ifft(fft_result)
        high_passed_signal = np.real(high_passed_signal)

        # Add the filtered signal to the new DataFrame
        filtered_df[column] = high_passed_signal

    # Add the last column (target variable) unchanged
    filtered_df[df.columns[-1]] = df[df.columns[-1]].values

    return filtered_df


def generate_orderbook_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates various features from order book data including spread, mid price,
    weighted prices, total depth on both sides, imbalance, and volatility of mid price.
    Automatically identifies ask/bid rate and size columns based on naming conventions.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the order book data.

    Returns:
    - pd.DataFrame: DataFrame with the original data and the newly generated features.
    """
    features_df = pd.DataFrame(index=df.index)

    # Identify columns
    ask_rate_cols = [col for col in df.columns if "askRate" in col]
    bid_rate_cols = [col for col in df.columns if "bidRate" in col]
    ask_size_cols = [col for col in df.columns if "askSize" in col]
    bid_size_cols = [col for col in df.columns if "bidSize" in col]

    # Fill NaN values with forward fill method to handle missing data
    df.fillna(method="ffill", inplace=True)

    # Calculate features
    features_df["spread"] = df[ask_rate_cols[0]] - df[bid_rate_cols[0]]
    features_df["mid_price"] = (df[ask_rate_cols[0]] + df[bid_rate_cols[0]]) / 2
    features_df["weighted_ask_price"] = sum(
        df[col] * df[ask_size_cols[i]] for i, col in enumerate(ask_rate_cols)
    ) / df[ask_size_cols].sum(axis=1)
    features_df["weighted_bid_price"] = sum(
        df[col] * df[bid_size_cols[i]] for i, col in enumerate(bid_rate_cols)
    ) / df[bid_size_cols].sum(axis=1)
    features_df["total_ask_size"] = df[ask_size_cols].sum(axis=1)
    features_df["total_bid_size"] = df[bid_size_cols].sum(axis=1)
    features_df["imbalance"] = (
        features_df["total_bid_size"] - features_df["total_ask_size"]
    ) / (features_df["total_bid_size"] + features_df["total_ask_size"])
    features_df["mid_price_volatility"] = (
        features_df["mid_price"].rolling(window=5).std()
    )
    features_df["y"] = df["y"]

    # Drop rows with NaN values created by rolling function
    features_df.dropna(inplace=True)

    return features_df


def normalize_train_test(
    train: pd.DataFrame, test: pd.DataFrame, stats_path: str = "feature_stats.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalizes all columns in the train and test DataFrames except for the last one using Z-score normalization
    based on the statistics (mean and std) from the training set, and saves these statistics to a CSV file.

    Parameters:
    - train (pd.DataFrame): The training DataFrame.
    - test (pd.DataFrame): The testing DataFrame.
    - stats_path (str): Path to save the feature statistics CSV file.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the normalized training and testing DataFrames,
                                         with all features except the last column normalized.
    """

    normalized_train = train.copy()
    normalized_test = test.copy()

    features_to_normalize = train.columns[:-1]
    stats = []

    for column in features_to_normalize:
        mean_val = train[column].mean()
        std_val = train[column].std()
        stats.append({"Feature": column, "Mean": mean_val, "Std": std_val})

        normalized_train[column] = (train[column] - mean_val) / std_val
        normalized_test[column] = (test[column] - mean_val) / std_val

    # Save statistics to CSV
    pd.DataFrame(stats).to_csv(stats_path, index=False)

    normalized_train.fillna(0, inplace=True)
    normalized_test.fillna(0, inplace=True)

    return normalized_train, normalized_test


def load_stats_and_normalize(
    data: pd.DataFrame, stats_path: str = "feature_stats.csv"
) -> pd.DataFrame:
    """
    Normalizes the DataFrame using Z-score normalization based on pre-computed feature statistics from a CSV file.

    Parameters:
    - data (pd.DataFrame): The DataFrame to be normalized.
    - stats_path (str): Path to the feature statistics CSV file.

    Returns:
    - pd.DataFrame: The normalized DataFrame.
    """
    normalized_data = data.copy()
    stats = pd.read_csv(stats_path)

    for _, row in stats.iterrows():
        feature, mean_val, std_val = row["Feature"], row["Mean"], row["Std"]
        normalized_data[feature] = (data[feature] - mean_val) / std_val

    normalized_data.fillna(0, inplace=True)

    return normalized_data


def apply_group_normalization(
    train: pd.DataFrame, test: pd.DataFrame, feature_groups: List[List[str]]
) -> None:
    """
    Applies group normalization for each group of features within the training dataset and
    applies the same transformation to the testing dataset. It asserts that all specified
    columns exist in the datasets.

    Parameters:
    - train (pd.DataFrame): The training dataset.
    - test (pd.DataFrame): The testing dataset.
    - feature_groups (List[List[str]]): A list of lists, where each inner list contains the names
                                        of the features that form a group to be normalized together.

    Returns:
    - None: The function modifies the training and testing DataFrames in place.
    """

    for feature_group in feature_groups:
        # Assert that all features in the group exist in the datasets
        for feature in feature_group:
            assert (
                feature in train.columns
            ), f"{feature} not found in training dataset columns"
            assert (
                feature in test.columns
            ), f"{feature} not found in testing dataset columns"

        # Calculate mean and std for the feature group in the training set
        group_mean = train[feature_group].dropna().values.mean()
        group_std = train[feature_group].dropna().values.std()

        # Normalize the training data
        for feature in feature_group:
            train[feature] = (
                train[feature].subtract(group_mean, axis=0).divide(group_std, axis=0)
            )

        # Apply the same transformation to the test data
        for feature in feature_group:
            test[feature] = (
                test[feature].subtract(group_mean, axis=0).divide(group_std, axis=0)
            )

    # Handle potential NaN values from division by zero
    train.fillna(0, inplace=True)
    test.fillna(0, inplace=True)


def differentiate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Differentiate all columns in a pandas DataFrame except for the last column,
    replacing each column with its first difference. This process is useful for
    time series analysis to make data more stationary by removing trends.

    Parameters:
    - df: pd.DataFrame, the input DataFrame with one or more columns.

    Returns:
    - pd.DataFrame: A new DataFrame where each column, except for the last one,
      has been replaced by its first difference. The last column remains unchanged.

    Note:
    The first row of the differentiated columns will contain NaN values due to
    the differencing process (as there's no preceding value for the first observation).
    """

    # Copy the DataFrame to avoid modifying the original data
    differentiated_df = df.copy()

    # Apply differencing to all columns except the last one
    for column in differentiated_df.columns[:-1]:  # Exclude the last column
        differentiated_df[column] = differentiated_df[column].diff().fillna(0)

    return differentiated_df


def apply_pca_keep_variance(
    df: pd.DataFrame, variance_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Apply Principal Component Analysis (PCA) to reduce the features of a DataFrame,
    excluding the last column, and retain only the components that represent at least
    a specified cumulative variance.

    Parameters:
    - df: pd.DataFrame, the input DataFrame with features.
    - variance_threshold: float, the proportion of variance to keep (between 0 and 1).

    Returns:
    - pd.DataFrame: A DataFrame where the original features (except the last column)
      are replaced with principal components that cumulatively explain at least the
      specified variance. The last column of the original DataFrame is appended to this
      DataFrame unchanged.

    Note:
    The function standardizes the data before applying PCA, as PCA is affected by
    scale. Ensure that all the columns to be transformed are numerical.
    """
    df = df.dropna()
    # Separate the DataFrame into the features to transform and the last column to exclude
    features_to_transform = df.iloc[:, :-1]
    last_column = df.iloc[:, -1]

    # Standardize the features (mean=0 and variance=1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_to_transform)

    # Apply PCA with variance threshold
    pca = PCA(n_components=variance_threshold)
    principal_components = pca.fit_transform(scaled_features)

    # Create a DataFrame with the principal components
    columns = [f"PC{i+1}" for i in range(principal_components.shape[1])]
    pca_df = pd.DataFrame(data=principal_components, columns=columns)

    # Append the last column from the original DataFrame to the PCA DataFrame
    pca_df = pd.concat([pca_df, last_column.reset_index(drop=True)], axis=1)

    return pca_df
