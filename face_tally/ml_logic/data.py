import pandas as pd
from face_tally.params import *
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
from keras_cv import bounding_box
from keras_cv import visualization
import keras_cv
import shutil

def load_annotations_csv():
    project_root = os.path.dirname(os.getcwd())

    path_annot = os.path.join(project_root, 'raw_data', 'bbox_train.csv')

    df = pd.read_csv(path_annot)
    return df

def load_image(image_path: str) -> tf.Variable:
    '''
    Function to load images without resizing, resize later
    '''
    # Read the image file and decode it to a tensor
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image
