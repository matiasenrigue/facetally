import numpy as np
import pandas as pd

from colorama import Fore, Style

def preprocess():
    """
    - Query the raw dataset from TO BE DETERMINED
    - Cache query result as a local CSV if it doesn't exist locally
    - Process query data
    - Store processed data on your personal BQ (truncate existing table if it exists)
    - No need to cache processed data as CSV (it will be cached when queried back from BQ during training)
    """
    pass


def train():
    """
    - Retrieve data from BigQuery, or from `cache_path` if the file exists
    - Store at `cache_path` if retrieved from BigQuery for future use
    - Train on the preprocessed dataset
    - Store training results and model weights

    Return loss as a float
    """
    pass


def evaluate():
    """
    Evaluate the performance of the latest production model on processed data
    Return loss as a float
    """
    pass


def pred():
    """
    Make a prediction using the latest trained model
    """
    pass


if __name__ == "__main__":
    # preprocess()
    # train()
    # evaluate()
    # pred()
    pass
