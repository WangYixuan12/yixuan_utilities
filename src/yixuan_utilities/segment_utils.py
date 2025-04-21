# Import necessary libraries
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def label_image(img: np.ndarray, predictor: SamPredictor) -> np.ndarray:
    """Interactive Image Labeling with SAM

    Label an image with a binary mask using SAM interactively using mouse clicks
    Left click: add positive point
    Right click: add negative point
    Middle click: remove point
    Implement this function using opencv mouse callback function

    Args:
        img (np.ndarray): The input image as a numpy array.
        predictor (SamPredictor): The SAM predictor object.

    Returns:
        np.ndarray: The binary mask as a numpy array.
    """
    # Initialize lists to store points and labels
    points = []
    labels = []
    masks = None

    # Process image with predictor
    predictor.set_image(img)

    # Mouse callback function
    def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal masks
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click
            points.append([x, y])
            labels.append(1)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click
            points.append([x, y])
            labels.append(0)
        elif event == cv2.EVENT_MBUTTONDOWN:  # Middle click
            if points:
                points.pop()
                labels.pop()

        # Update masks after each click
        if points:
            input_points = np.array(points)
            input_labels = np.array(labels)
            masks, scores, logits = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

    # Set up OpenCV window and set mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        # Display the image
        display_img = img.copy()
        for point, label in zip(points, labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_img, tuple(point), 5, color, -1)

        # Overlay the mask if available
        if masks is not None:
            mask_overlay = (masks[0] * 255).astype(np.uint8)
            display_img = cv2.addWeighted(
                display_img, 0.7, cv2.cvtColor(mask_overlay, cv2.COLOR_GRAY2BGR), 0.3, 0
            )

        cv2.imshow("Image", display_img)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

    # Return the first mask (or handle multiple masks as needed)
    return masks[0] if masks is not None else None


def load_sam() -> SamPredictor:
    """Load the SAM model and create a predictor.

    Returns:
        SamPredictor: The SAM predictor object.
    """
    curr_path = os.path.dirname(os.path.abspath(__file__))
    Path(curr_path).mkdir(parents=True, exist_ok=True)
    sam_checkpoint = f"{curr_path}/ckpts/sam_vit_h_4b8939.pth"
    remote_sam = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    if not os.path.exists(sam_checkpoint):
        os.system(f"wget {remote_sam} -P {curr_path}/ckpts/")
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor


if __name__ == "__main__":
    # Example usage
    curr_path = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(f"{curr_path}/sam_test.jpg")
    predictor = load_sam()
    mask = label_image(img, predictor)
    print("Generated mask shape:", mask.shape)
