from tensorflow import keras
import keras_cv
from google.cloud import storage
from face_tally.params import *


def save_model(model_path, google_auth_credentials) -> None:
    model_filename = model_path.split("/")[-1]
    client = storage.Client(credentials=google_auth_credentials)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"models/{model_filename}")
    blob.upload_from_filename(model_path)

    print("✅ Weights saved to GCS")

    return None


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    """
    Model evaluation with COCO Metric Callback
    """

    def __init__(self, data, google_auth_credentials):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format=BOX_FORMAT,
            evaluate_freq=1e9,
        )

        self.best_map = -1.0
        self.google_auth_credentials = google_auth_credentials

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images = batch["images"]
            bounding_boxes = batch["bounding_boxes"]

            # Extract "boxes" and "classes" from bounding_boxes
            classes = bounding_boxes["classes"]
            boxes = bounding_boxes["boxes"]

            y_pred = self.model.predict(images, verbose=0)

            # Convert classes and bounding_boxes to a dictionary
            y_true = {"classes": classes, "boxes": boxes}

            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        current_map = metrics["map"]

        if current_map > self.best_map:
            self.best_map = current_map
            self.model.save(self.save_path)
            from_path = f"{self.save_path}_{current_map}_model_weights.h5"
            self.model.save_weights(from_path)
            save_model(from_path, self.google_auth_credentials)

        return logs
