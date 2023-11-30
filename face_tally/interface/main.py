import numpy as np
import pandas as pd

from colorama import Fore, Style
from face_tally.ml_logic.data import *
from face_tally.ml_logic.preprocessing import *
from face_tally.ml_logic.train import *


def preprocess():
    """
    - Query the raw dataset from Google CloudStorage
    - Download images locally
    - Process data
    """

    df = load_annotations_csv()

    df_normalized = normalize_data(df)

    dataset = create_dataset(df_normalized)

    return dataset


def train(data):
    """
    - Train on the preprocessed dataset
    - Store training results and model weights
    - Return loss as a float
    """

    # Comment: PENDING
    train_ds, val_ds, test_data = splitting_data(data)

    fitted_model = fit_model(train_ds, val_ds)

    return train_ds, val_ds, test_data, fitted_model


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
    dataset = preprocess()
    train_ds, val_ds, test_data = splitting_data(dataset)
    # evaluate()
    # pred()
    pass
