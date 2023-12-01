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
