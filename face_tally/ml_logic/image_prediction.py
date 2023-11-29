from ultralytics import YOLO
import numpy as np
import datetime

from PIL import Image
from pillow_heif import register_heif_opener
import cv2


"""
This script has functions to be able to predict where object are in a picture:
- getting_bounding_boxes will use the model to predict what objects are found and where they are
- create_image will create an image putting togheter both the original picture and the prediction
- save_image will save the image when the code is runned localy
- full_process will run the 3 functions
"""


def predict_bounding_boxes(image, model) -> dict:
    """
    Gets the bounding boxes that say where objects are located in an images, returns:
    - class of the object
    - position in the image
    - confidence of that object being it

    Does that by using our models prediction
    """

    results = model.predict(image)

    result = results[0]
    box = result.boxes[0]

    bound_boxes = []

    for box in result.boxes:
        cordenadas_xywh = box.xyxy[0].tolist()

        predicted_class = box.cls[0]
        class_name = result.names[predicted_class.item()]

        confidence = round(box.conf[0].item(), 2)

        dict = {
            "Object type": class_name,
            "Coordinates": cordenadas_xywh,
            "Probability": confidence,
        }

        bound_boxes.append(dict)

    return bound_boxes


def create_image(original_image_array: np.array, bound_boxes: dict) -> np.array:
    """
    Takes both:
    - The original image array
    - The result from the bounding boxes

    And returns an image with both elements in array format
    """

    # Create an OpenCV image from the numeric array
    opencv_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2BGR)

    # Annotate bounding boxes on the OpenCV image
    for box_info in bound_boxes:
        coordinates = box_info["Coordinates"]
        object_type = box_info["Object type"]
        probability = box_info["Probability"]

        coordinates = box_info["Coordinates"]
        color = (135, 206, 250)  # Color for the bounding box
        thickness = 5

        # Convert float coordinates to integers
        coordinates = [int(coord) for coord in coordinates]

        # Draw rectangle on the image
        cv2.rectangle(
            opencv_image,
            (coordinates[0], coordinates[1]),
            (coordinates[2], coordinates[3]),
            color,
            thickness,
        )

        # Annotate with object type and probability
        label = f"{object_type} ({probability:.2f})"
        cv2.putText(
            opencv_image,
            label,
            (coordinates[0], coordinates[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            4,
        )
        # cv2.FONT_HERSHEY_SIMPLEX, size, color, width

    # Convert the annotated image back to RGB format
    annotated_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)

    # Display or save the annotated image as needed
    return annotated_image


def save_image(image_created, file_save_name=None) -> None:
    """
    - Uses the output of the create_image function to save the image or display it
    - Useless unless the code is being runned in the local environment
    """

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if file_save_name is None:
        save_path = f"done_images/picture_{current_time}.jpg"
    else:
        save_path = f"done_images/picture_{file_save_name}_{current_time}.jpg"

    Image.fromarray(image_created).save(save_path)
    print(f"Image saved at: {save_path}")


def full_process(original_image, model, saving_name=None) -> None:
    """
    Calls the 3 functions
    """

    array_original_image = np.array(original_image)

    bbs = predict_bounding_boxes(original_image, model)
    created_image = create_image(array_original_image, bbs)

    save_image(created_image, saving_name)

    Image.fromarray(created_image)


if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    image_file_path = "trial_images/IMG_4194.HEIC"

    register_heif_opener()  # In case if its an iphone picture

    image = Image.open(image_file_path)
    array_image = np.array(image)

    full_process(image, model, "prueba_now")
