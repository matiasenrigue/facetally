from tensorflow import keras
import keras_cv
from google.cloud import storage
from face_tally.params import *


def save_model_GCP(model_path) -> None:
    model_filename = model_path.split("/")[-1]
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Weights saved to GCS")

    return None


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    """
    Model evaluation with COCO Metric Callback
    """

    def __init__(self, data, best_MaP):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format=BOX_FORMAT,
            evaluate_freq=1e9,
        )
        self.best_map = best_MaP

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images = batch[0]
            bounding_boxes = batch[1]

            # Extract "boxes" and "classes" from bounding_boxes
            classes = bounding_boxes["classes"]
            boxes = bounding_boxes["boxes"]
            y_pred = self.model.predict(images, verbose=0)

            # Convert classes and bounding_boxes to a dictionary
            y_true = {"classes": classes, "boxes": boxes}

            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result()

        logs.update(metrics)

        current_map = metrics["MaP"]

        if current_map > self.best_map:
            self.best_map = current_map
            model_path = os.path.join(LOCAL_DATA_PATH, "models")
            os.makedirs(model_path, exist_ok=True)

            from_path = os.path.join(model_path, f"yolo_{current_map}_weights.h5")
            self.model.save_weights(from_path)
            save_model_GCP(from_path)

        return logs
