import tensorflow as tf
from keras_cv import models
from face_tally.params import *
from face_tally.credentials import create_google_cloud_client
import comet_ml
from comet_ml import API


def get_yolo() -> models.YOLOV8Backbone:
    """
    Define YOLO model with face class and COCO backbone
    Need to define the size of the backbone
    """
    # Size options: xs, s, m, l, xl
    backbone_name = "yolo_v8_" + str(BACKBONE_SIZE) + "_backbone_coco"

    yolo = models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format=BOX_FORMAT,
        backbone=models.YOLOV8Backbone.from_preset(backbone_name),
        fpn_depth=1,
    )

    return yolo


async def download_best_model_from_GCP(
    bucket_name: str,
    folder_path: str,
) -> bool:
    """
    Given a BUCKET_NAME, it updated the data in bucket folder path to a local destination folder
    If overwrite True, the data from the bucket overwrites the local data
    Returns True if something is downloaded or updated
    """
    # Initialize the Cloud Storage client
    storage_client = await create_google_cloud_client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # Check if there is any existing blob in the folder. Return None if not
    # Count the number of blobs in the folder, excluding the folder placeholder
    blobs = bucket.list_blobs(prefix=folder_path)
    blob_count = sum(1 for blob in blobs if blob.name != folder_path)
    if blob_count == 0:
        return None, -1

    # List objects in the specified folder path
    blobs = bucket.list_blobs(prefix=folder_path)

    # Sort all file in numerical order from smaller to bigger
    # Expected file name format: models/yolo_MaP_weights.h5 (where Map is a float)
    all_files = [each.name for each in blobs if each.name != "models/"]
    all_files.sort(key=lambda x: float(x.split("_")[1]))

    # Select the best model (higher MaP)
    best_blob = all_files[-1]
    print("Using best model for tarining: ", best_blob)

    # Extract best Map
    MaP = float(best_blob.split("_")[1])

    # Download the model
    blob = bucket.blob(best_blob)

    # Define the local folder to save the model
    weights_path = os.path.join(LOCAL_DATA_PATH, "models")

    # Create the local folder if it doesn't exist
    os.makedirs(weights_path, exist_ok=True)
    save_path = os.path.join(weights_path, "best_weight.h5")

    # Download model
    blob.download_to_filename(save_path)

    # Load saved model
    model = get_yolo()
    model.load_weights(save_path)

    return model, MaP


async def get_model():
    """
    Get yolo model from GCP or from backbone if there is no model available
    """
    bucket_model_folder = "models/"

    model, MaP = await download_best_model_from_GCP(
        BUCKET_NAME,
        bucket_model_folder,
    )
    if model is None:
        # Load model from backbone trained with coco
        print("Loading model from backbone, size: ", str(BACKBONE_SIZE))
        model = get_yolo()

    return model, MaP


def compile_model(model: models.YOLOV8Detector) -> models.YOLOV8Detector:
    """
    Compile the model with binary_crossentropy and Complete IoU (CIoU) metric.
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    model.compile(
        optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
    )
    return model
