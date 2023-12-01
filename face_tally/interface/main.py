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
    print("Starting preprocessing")

    df = load_annotations_csv()

    df_normalized = normalize_data(df)

    dataset = create_dataset(df_normalized)

    return dataset


def train(data):
    """
    - Train on the preprocessed dataset
    - Store model weights
    - Return ???????
    """

    print("Starting training")

    # Split dataset to train, test, val
    train_ds, val_ds, test_data = splitting_data(data)

    # Adapt input for fitting
    train_ds = dict_to_tuple_ds(train_ds)
    val_ds = dict_to_tuple_ds(val_ds)

    # Fit the model
    breakpoint()
    yolo, history = fit_model(train_ds, val_ds)

    print("Training done")

    return yolo, history, test_data


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
    # Updates the local raw data with the data in Google Cloud Storage
    update_local_raw_data_from_GCP()

    # Preprocess the data and convert it to yolo format
    dataset = preprocess()

    # Train the yolo model with the preprocessed data
    yolo, history, test_data = train(dataset)
