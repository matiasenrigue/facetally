import tensorflow as tf
from keras_cv import models
from face_tally.params import *
from google.cloud import storage


def get_yolo() -> models.YOLOV8Backbone:
    """
    Define YOLO model with face class and COCO backbone
    Need to define the size of the backbone
    """
    # Size options: xs, s, m, l, xl
    backbone_name = "yolo_v8_" + BACKBONE_SIZE + "_backbone_coco"

    yolo = models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format=BOX_FORMAT,
        backbone=models.YOLOV8Backbone.from_preset(backbone_name),
        fpn_depth=1,
    )
    return yolo


def download_best_model_from_GCP(
    bucket_name: str,
    folder_path: str,
) -> bool:
    """
    Given a BUCKET_NAME, it updated the data in bucket folder path to a local destination folder
    If overwrite True, the data from the bucket overwrites the local data
    Returns True if something is downloaded or updated
    """
    # Initialize the Cloud Storage client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List objects in the specified folder path
    blobs = bucket.list_blobs(prefix=folder_path)

    # Sort all file in numerical order from smaller to bigger
    all_files = [each.name for each in blobs if each.name != "models/"]
    all_files.sort(key=lambda x: float(x.split("_")[1]))

    # Select the best model (higher MaP)
    best_blob = all_files[-1]

    # Download the model
    blob = bucket.blob(best_blob)

    weights_path = os.path.join(LOCAL_DATA_PATH, "models")

    # Create a local folder if it doesn't exist
    os.makedirs(weights_path, exist_ok=True)
    save_path = os.path.join(weights_path, "best_weight.h5")
    blob.download_to_filename(save_path)

    # Load saved model
    model = models.load_model(save_path)

    return model


def get_model():
    """
    Get yolo model from GCP or from backbone if there is no model available
    """
    bucket_model_folder = "models/"

    model = download_best_model_from_GCP(
        BUCKET_NAME,
        bucket_model_folder,
    )
    if model is None:
        # Load model from backbone trained with coco
        print("Loading model from backbone")
        model = get_yolo()

    return model


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
