import numpy as np
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
