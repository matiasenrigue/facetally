import pandas as pd
from face_tally.params import *
import os
import pandas as pd
import tensorflow as tf
from google.cloud import storage


def load_annotations_csv():
    """
    This function is to load the annotations of the bboxes (see in preprocessing)
    """
    project_root = os.path.dirname(os.getcwd())

    path_annot = os.path.join(project_root, "raw_data", "bbox_train.csv")

    df = pd.read_csv(path_annot)
    return df


def load_image(image_path: str) -> tf.Variable:
    """
    This function is to load images without resizing, resize later
    """
    # Read the image file and decode it to a tensor
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image


def download_images_from_GCP(
    bucket_name: str, folder_path: str, destination_folder: str, overwrite=False
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
    folder_objects = bucket.list_blobs(prefix=folder_path)

    # Create a local folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    # Download each image file from the folder
    count = 0
    for obj in folder_objects:
        # Construct local file path for downloading
        local_file_path = os.path.join(destination_folder, os.path.basename(obj.name))
        # Check if the file exists locally
        if overwrite or not os.path.exists(local_file_path):
            # Download the object to the local file
            obj.download_to_filename(local_file_path)
            count += 1
            print(f"Downloaded '{obj.name}' to '{local_file_path}'.  Count: {count}")

    return True if count > 0 else False
