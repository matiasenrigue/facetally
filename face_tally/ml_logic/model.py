from ultralytics import YOLO


def train_model():
    # Load the pre-trained model
    model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")

    # Train the model
    model.train(
        data="dataset.yaml", epochs=200, imgsz=256, save=True, format="onnx"
    )  # Set imgsz to 256 for training on 256x256 images

    # Export the model to ONNX format
    path = model.export()
    print(path)
