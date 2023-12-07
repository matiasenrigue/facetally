import pandas as pd
from face_tally.params import *
import os
import pandas as pd
import tensorflow as tf
from google.cloud import storage
from face_tally.credentials import create_google_cloud_client
import asyncio
import zipfile


def load_annotations_csv() -> pd.DataFrame:
    """
    This function is to load the annotations of the bboxes (see in preprocessing)
    """
    path_annot = os.path.join(LOCAL_DATA_PATH, "bbox_train.csv")

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


async def download_data_from_GCP(
    bucket_name: str, folder_path: str, destination_folder: str, overwrite=False
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


def unzip_file(zip_path: str, destination_folder: str) -> None:
    """
    Unzips the zip under zip_path and saves the content into destination_folder
    """
    print("Unzipping images...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            file_path = os.path.join(destination_folder, file_info.filename)

            # Check if the file already exists, and if so, overwrite it
            # if os.path.exists(file_path):
                # os.rmdir(file_path)

            zip_ref.extract(file_info, destination_folder)

    return None


async def update_local_raw_data_from_GCP() -> None:
    """
    Updates the local raw data with the data in Google Cloud Storage
    """

    print("Updating local raw data from Google Cloud Storage...")

    # Bucket paths
    image_data_zip_name = "image_data.zip"
    csv_name = "bbox_train.csv"

    # Download the data
    changes_csv = await download_data_from_GCP(BUCKET_NAME, csv_name, LOCAL_DATA_PATH)
    changes_images = await download_data_from_GCP(
        BUCKET_NAME, image_data_zip_name, LOCAL_DATA_PATH
    )

    # Unzip the data
    zip_path = os.path.join(LOCAL_DATA_PATH, image_data_zip_name)
    unzip_file(zip_path, LOCAL_DATA_PATH)

    if not (changes_csv or changes_images):
        print("Process finished, local files were already up to date")
    else:
        print("Process finished, local raw data folder is up to date")

    return None

async def update_template_images_from_GCP() -> None:
    """
    Updates the template images with the data in Google Cloud Storage
    """

    print("Updating local images from Google Cloud Storage...")

    # Bucket paths
    image_data_zip_name = "images.zip"

    # Download the data
    changes= await download_data_from_GCP(BUCKET_NAME, image_data_zip_name, LOCAL_DATA_PATH)

    # Unzip the data
    zip_path = os.path.join(LOCAL_DATA_PATH, image_data_zip_name)
    unzip_file(zip_path, LOCAL_DATA_PATH)

    if not (changes):
        print("Process finished, local files were already up to date")
    else:
        print("Process finished, local raw data folder is up to date")

    return None

if __name__ == "__main__":
    asyncio.run(update_local_raw_data_from_GCP())
