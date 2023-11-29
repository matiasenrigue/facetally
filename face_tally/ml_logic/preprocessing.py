from face_tally.ml_logic.data import *
import pandas as pd
import tensorflow as tf
import os

"""
SCRIP DOCU PENDING
"""


def aggregate_boxes(data):
    """
    This function does....
    """
    boxes = data[["LEFT", "TOP", "RIGHT", "BOTTOM"]].values.tolist()
    return boxes


def normalize_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    DOCUMENTATION PENDING
    """
    df["LEFT"] = df["xmin"] / df["width"]
    df["TOP"] = df["ymin"] / df["height"]
    df["RIGHT"] = df["xmax"] / df["width"]
    df["BOTTOM"] = df["ymax"] / df["height"]

    # Selecting only the required columns
    df_rel_xyxy = df[["Name", "LEFT", "TOP", "RIGHT", "BOTTOM"]]

    # Apply the aggregate function to the grouped data and reset the index.
    grouped = (
        df_rel_xyxy.groupby("Name").apply(aggregate_boxes).reset_index(name="boxes")
    )

    return grouped


def add_image_path_to_bbox(df: pd.DataFrame) -> tf.data.Dataset:
    """
    The dataframe we are inputting is the grouped dataframe created with normalize_data function
    OUTPUT DOUC PENDING
    """
    image_paths = []
    bboxes = []
    project_root = os.path.dirname(os.getcwd())
    test_image_folder = os.path.join(project_root, "raw_data", "image_data")
    # Iterate through the grouped dataframe and populate the lists with image paths and bounding boxes.
    for _, row in df.iterrows():
        image_path = os.path.join(test_image_folder, row["Name"])
        image_bboxes = row["boxes"]
        image_paths.append(image_path)
        bboxes.append(image_bboxes)

    # Convert the lists to TensorFlow tensors. Use a ragged tensor for bounding boxes to handle varying lengths.
    bbox = tf.ragged.constant(bboxes, dtype=tf.float32)  # Bounding boxes
    classes = tf.ragged.constant(
        [[0] * len(b) for b in bboxes], dtype=tf.int32
    )  # Class labels
    image_paths = tf.constant(image_paths)  # Image paths

    # Create a tf.data.Dataset that combines the image paths with bounding boxes and class labels
    data = tf.data.Dataset.from_tensor_slices((image_paths, bbox, classes))

    return data


def load_dataset(image_path: str, bbox: tf.Variable, classes: tf.Variable) -> dict:
    """
    Function to create the data dictionary required by KerasCV without resizing the image
    """
    # Load the image
    image = load_image(image_path)
    # Create a dictionary for bounding boxes with 'boxes' and 'classes' as keys
    bounding_boxes = {"boxes": bbox, "classes": classes}
    # Return a dictionary with 'images' and 'bounding_boxes'
    return {"images": image, "bounding_boxes": bounding_boxes}
