import tensorflow as tf
from keras_cv import models
from face_tally.params import *


def get_yolo(bbox_format=BOX_FORMAT):
    """
    Define YOLO model with face class and COCO backbone
    """
    yolo = models.YOLOV8Detector(
        num_classes=len(class_mapping),
        bounding_box_format=bbox_format,
        backbone=models.YOLOV8Backbone.from_preset("yolo_v8_s_backbone_coco"),
        fpn_depth=1,
    )
    return yolo


def get_model():
    """
    DOCUMENTATION
    """

    # PENDING
    # load_best_model_from_GCP()
    # Load list of model weigths from the GS - sort this list by metrics and load best ones
    # bestmodel

    # Get yolo model from GCP or from backbone if there is no model available
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
