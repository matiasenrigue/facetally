from face_tally.ml_logic.data import *
from face_tally.ml_logic.preprocessing import *
from face_tally.ml_logic.train import splitting_data
from face_tally.ml_logic.model import *
import asyncio


async def evaluate():
    """
    - Evaluate the trained model on the test data set
    - Output evaluation metrics, such as Mean Average Precision
    """

    # Updates the local raw data with the data in Google Cloud Storage
    await update_local_raw_data_from_GCP()

    # Preprocess the data and convert it to yolo format
    print("Starting preprocessing")

    df = load_annotations_csv()

    df_normalized = normalize_data(df)

    dataset = create_dataset(df_normalized)

    print("✅ Preprocessing done")

    print("Starting evaluation")

    # Split dataset to train, test, val
    _, _, test_data = splitting_data(dataset)
    map_score = await evaluate_model(test_data)

    print(f"Model evaluation completed. MaP: {map_score}")


if __name__ == "__main__":
    asyncio.run(evaluate())
