from keras_cv import bounding_box
from keras_cv import visualization
from face_tally.params import *
from face_tally.interface.main import preprocess, train
from face_tally.ml_logic.train import splitting_data
import matplotlib.pyplot as plt


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    """
    This function is meant to visualise our preprocessed training data
    """
    inputs = next(iter(inputs.take(1)))

    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    figure = visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


# NOTES: TO USE FIRST FUNCTION AND VISUALISE DATASET


def test_preprocessing():
    dataset = preprocess()

    train_ds, val_ds, test_data = splitting_data(dataset)

    visualize_dataset(
        train_ds, bounding_box_format=BOX_FORMAT, value_range=(0, 255), rows=2, cols=2
    )


#####################################################################################################
#####################################################################################################


def visualize_detections(model, dataset, bounding_box_format):
    """
    This function is meant to visualise our predictions
    """
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    class_mapping = {0: "face"}
    figure = visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=2,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )


# NOTES: TO USE SECOND FUNCTION


def test_training():
    dataset = preprocess()
    train_ds, val_ds, test_data = splitting_data(dataset)
    yolo = train(dataset)

    visualize_detections(yolo, dataset=val_ds, bounding_box_format=BOX_FORMAT)


if __name__ == "__main__":
    test_preprocessing()
    test_training()
