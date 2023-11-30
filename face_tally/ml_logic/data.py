import pandas as pd
from face_tally.params import *
import os
import pandas as pd
import tensorflow as tf


def load_annotations_csv():
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
