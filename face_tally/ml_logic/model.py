import tensorflow as tf
from keras_cv import models
from face_tally.params import *
from face_tally.credentials import create_google_cloud_client
import comet_ml
from comet_ml import API
from ultralytics import YOLO


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
    print("✅ Loaded weights from GCP: ", best_blob)

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


def download_best_model_from_Comet():
    """
    Connects to Comet and downloads the best model (Production)
    Weights saved loacally, then loaded into YOLO model
    """
    # Initialize Comet ML API connection
    api = API()
    comet_ml.init()

    # Try to use pretrained weights if available
    try:
        # Fetching the model from Comet ML
        models = api.get_model(
            workspace=COMET_WORKSPACE_NAME,
            model_name=COMET_MODEL_NAME,
        )

        # Get production model weights
        model_versions = models.find_versions(status="Production")
        latest_production_weights = model_versions[0]

        # Preparing local path for weights
        weights_path = os.path.join(LOCAL_DATA_PATH, "models_Comet")
        os.makedirs(weights_path, exist_ok=True)

        # Downloading the weights
        models.download(
            version=latest_production_weights,
            output_folder=weights_path,
            expand=True,
        )

        # Load the model with the downloaded weights
        model = YOLO(os.path.join(weights_path, "best.pt"))
        print("✅ Loaded weights from the comet ML")

        return model

    # If loading pretrained weights fails, initialize a new model
    except Exception as error:
        print(f"❌ Could not load weights from Comet: {error}")
        return None


async def get_model(source=MODEL_SOURCE):
    """
    Get yolo model from source or from backbone if there is no model available
    Available sources: GCP and Comet. The output model is different
    """
    model = None
    MaP = -1

    if source == "GCP":
        bucket_model_folder = "models/"
        model, MaP = await download_best_model_from_GCP(
            BUCKET_NAME,
            bucket_model_folder,
        )
    elif source == "COMET":
        model = download_best_model_from_Comet()

    if model is None:
        # Load model from backbone trained with coco
        print("Loading model from backbone, size: ", str(BACKBONE_SIZE))
        model = get_yolo()

    return model, MaP


async def get_model_for_training():
    """
    Get yolo model from GCP or from backbone if there is no model available
    """
    model, MaP = await get_model("GCP")

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
