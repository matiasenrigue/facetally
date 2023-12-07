from face_tally.ml_logic.data import *
from face_tally.ml_logic.preprocessing import *
from face_tally.ml_logic.train import splitting_data
from face_tally.ml_logic.model import *
import asyncio
from keras_cv.metrics import BoxCOCOMetrics


async def evaluate_model(test_ds, source):
    """
    Evaluate the provided YOLO model on the given test data.
    """
    # Get the model
    model, _ = await get_model(source=source)
    map_score = -1

    if source == "GCP":
        # Initialize COCO metrics
        coco_metrics = BoxCOCOMetrics(bounding_box_format=BOX_FORMAT, evaluate_freq=1)

        # Evaluate each image in the test data
        for batch in test_ds:
            images = batch[0]
            bounding_boxes = batch[1]

            classes = bounding_boxes["classes"]
            boxes = bounding_boxes["boxes"]

            # Make predictions
            y_pred = model.predict(images)

            # Prepare ground truth data (y_true)
            y_true = {"boxes": boxes, "classes": classes}

            # Update metrics
            coco_metrics.update_state(y_true, y_pred)

        # Calculate the final result
        results = coco_metrics.result()
        map_score = results["MaP"]  # Make sure 'MaP' is a key in the results dictionary

    elif source == "COMET":
        pass

    print(f"Mean Average Precision (MaP) of the model on test data: {map_score}")

    return map_score


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
    map_score = await evaluate_model(test_data, MODEL_SOURCE)

    print(f"Model evaluation completed. MaP: {map_score}")


if __name__ == "__main__":
    asyncio.run(evaluate())
