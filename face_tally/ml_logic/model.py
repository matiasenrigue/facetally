from ultralytics import YOLO
import tensorflow as tf
import keras
from tensorflow import keras
from keras_cv import bounding_box, visualization, layers, models
from face_tally.params import *


def get_backbone():
    """
    We will use yolov8 small backbone with coco weights
    """
    backbone = models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco")
    return backbone


def get_yolo(bbox_format="rel_xyxy"):
    """
    DOCUMENTATION PENDING
    """
    class_mapping = {0: "face"}
    backbone = get_backbone()
    yolo = models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format=bbox_format,
        backbone=backbone,
        fpn_depth=1,
    )
    return yolo


def compile():
    """
    DOCUMENTATION PENDING
    """
    yolo = get_yolo()
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        global_clipnorm=GLOBAL_CLIPNORM,
    )
    return yolo.compile(
        optimizer=optimizer, classification_loss="binary_crossentropy", box_loss="ciou"
    )
