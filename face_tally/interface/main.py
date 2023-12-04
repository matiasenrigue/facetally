from colorama import Fore, Style
from face_tally.ml_logic.data import *
from face_tally.ml_logic.preprocessing import *
from face_tally.ml_logic.train import *
import asyncio


def preprocess():
    """
    - Query the raw dataset from Google Cloud Storage
    - Download images locally
    - Process data into yolo format, save in RAM only
    """
    print("Starting preprocessing")

    df = load_annotations_csv()

    df_normalized = normalize_data(df)

    dataset = create_dataset(df_normalized)

    print("✅ Preprocessing done")

    return dataset


async def train(data):
    """
    - Train on the preprocessed dataset
    - Store model weights locally and in Google Cloud Storage
    - Return model and split test data set
    """

    print("Starting training")

    # Split dataset to train, test, val
    train_ds, val_ds, test_data = splitting_data(data)

    # Fit the model
    yolo, history = await fit_model(train_ds, val_ds)

    print("✅ Training done")

    return yolo, test_data


async def main():
    # Updates the local raw data with the data in Google Cloud Storage
    await update_local_raw_data_from_GCP()

    # Preprocess the data and convert it to yolo format
    dataset = preprocess()

    # Train the yolo model with the preprocessed data
    yolo, test_data = await train(dataset)


if __name__ == "__main__":
    asyncio.run(main())
