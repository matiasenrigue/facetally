from face_tally.params import *
from face_tally.ml_logic.preprocessing import load_dataset
from face_tally.ml_logic.model import *
import tensorflow as tf
from tensorflow import keras
from keras_cv import layers


def get_augmenter(bbox_format="rel_xyxy"):
    """
    Function to define our augmenter, will be applied on train_ds
    """
    augmenter = keras.Sequential(
        layers=[
            layers.RandomFlip(mode="horizontal", bounding_box_format=bbox_format),
            layers.RandomShear(
                x_factor=0.2, y_factor=0.2, bounding_box_format=bbox_format
            ),
            layers.JitteredResize(
                target_size=(640, 640),
                scale_factor=(0.75, 1.3),
                bounding_box_format=bbox_format,
            ),
        ]
    )
    return augmenter


def get_resizer(bbox_format="rel_xyxy"):
    """
    Function to define our resizer, will be applied on val_ds
    """
    resizing = layers.JitteredResize(
        target_size=(640, 640),
        scale_factor=(0.75, 1.3),
        bounding_box_format=bbox_format,
    )
    return resizing


def splitting_data(data: tf.data.Dataset):
    """
    Function to split the data into train, validation, and test datasets (80%, 15%, 5%)
    Applies augmenter to train_ds and resizer to validation_ds
    """
    augmenter = get_augmenter()
    resizing = get_resizer()

    all_images_len = data.cardinality().numpy()

    train_idx = int(all_images_len * 0.8)
    validation_idx = int(all_images_len * 0.15)

    train_data = data.take(train_idx)
    val_data = data.skip(train_idx).take(validation_idx)
    test_data = data.skip(train_idx + validation_idx)

    # Just for test
    train_data = train_data.take(4)

    train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    val_ds = val_ds.map(resizing, num_parallel_calls=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_data


def dict_to_tuple(inputs):
    """
    Defines the class ids and mapping (in this case, we only have one class "face")
    """
    return inputs["images"], inputs["bounding_boxes"]


def dict_to_tuple_train(train_ds):
    """
    Applying dict_to_tuple function on train_ds
    """
    train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    return train_ds


def dict_to_tuple_val(val_ds):
    """
    Applying dict_to_tuple function on val_ds
    """
    val_ds = val_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    return val_ds


def fit_model(train_ds, val_ds):
    """
    Fitting model on train_ds, using val_ds as validation data
    """
    yolo = get_yolo()
    yolo_compile = get_compile(yolo)
    train_ds = dict_to_tuple_train(train_ds)
    val_ds = dict_to_tuple_val(val_ds)
    history = yolo_compile.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
    return history
