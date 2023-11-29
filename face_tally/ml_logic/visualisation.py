from keras_cv import bounding_box
from keras_cv import visualization


def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    """
    DOCUMENTATION PENDING
    """
    inputs = next(iter(inputs.take(1)))
    class_mapping = {0: "face"}
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
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

# visualize_dataset(
#     train_ds, bounding_box_format="rel_xyxy", value_range=(0, 255), rows=2, cols=2
# )

# visualize_dataset(
#     val_ds, bounding_box_format="rel_xyxy", value_range=(0, 255), rows=2, cols=2
# )


def visualize_detections(model, dataset, bounding_box_format):
    """
    DOCUMENTATION PENDING
    """
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    class_mapping = {0: "face"}
    visualization.plot_bounding_box_gallery(
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

# visualize_detections(yolo, dataset=val_ds, bounding_box_format="rel_xyxy")
