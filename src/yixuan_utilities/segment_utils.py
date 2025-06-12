# Import necessary libraries
import os
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from XMem.dataset.range_transform import im_normalization
from XMem.inference.data.mask_mapper import MaskMapper
from XMem.inference.inference_core import InferenceCore
from XMem.model.network import XMem


@torch.no_grad()
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

def load_sam_mask_generator(*args, **kwargs) -> SamAutomaticMaskGenerator:
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
    mask_generator = SamAutomaticMaskGenerator(sam, *args, **kwargs)
    return mask_generator


def load_XMem() -> Union[InferenceCore, MaskMapper, T.Compose, T.Compose]:
    curr_path = os.path.dirname(os.path.abspath(__file__))
    Path(curr_path).mkdir(parents=True, exist_ok=True)
    xmem_ckpt_path = f"{curr_path}/ckpts"
    xmem_ckpt_name = "XMem.pth"
    xmem_ckpt = f"{xmem_ckpt_path}/{xmem_ckpt_name}"
    device = "cuda"
    if not os.path.exists(xmem_ckpt):
        print("Downloading XMem model...")
        os.system(
            f"wget -P {xmem_ckpt_path} https://github.com/hkchengrex/XMem/releases/download/v1.0/{xmem_ckpt_name}"
        )
    xmem_config = {
        "model": xmem_ckpt,
        "disable_long_term": False,
        "enable_long_term": True,
        "max_mid_term_frames": 10,
        "min_mid_term_frames": 5,
        "max_long_term_elements": 10000,
        "num_prototypes": 128,
        "top_k": 30,
        "mem_every": 5,
        "deep_update_every": -1,
        "save_scores": False,
        "size": 480,
        "key_dim": 64,
        "value_dim": 512,
        "hidden_dim": 64,
        "enable_long_term_count_usage": True,
    }

    network = XMem(xmem_config, xmem_config["model"]).to(device).eval()
    model_weights = torch.load(xmem_config["model"])
    network.load_weights(model_weights, init_as_zero_if_needed=True)

    xmem_processor = InferenceCore(network, config=xmem_config)
    xmem_mapper = MaskMapper()
    xmem_im_transform = T.Compose(
        [
            T.ToTensor(),
            im_normalization,
            T.Resize(xmem_config["size"], interpolation=T.InterpolationMode.BILINEAR),
        ]
    )
    xmem_mask_transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(xmem_config["size"], interpolation=T.InterpolationMode.NEAREST),
        ]
    )
    return xmem_processor, xmem_mapper, xmem_im_transform, xmem_mask_transform


@torch.no_grad()
def track_mask(
    rgb: np.ndarray,
    mask: Optional[np.ndarray],
    xmem_processor: InferenceCore,
    xmem_mapper: MaskMapper,
    xmem_im_transform: T.Compose,
    xmem_mask_transform: T.Compose,
) -> np.ndarray:
    """Track the mask in the video stream.

    Args:
        rgb (np.ndarray): The input RGB image.
        mask (np.ndarray): The binary mask to track.
        xmem_processor (InferenceCore): The XMem processor object.
        xmem_mapper (MaskMapper): The XMem mask mapper object.
        xmem_im_transform (T.Compose): The image transformation pipeline.
        xmem_mask_transform (T.Compose): The mask transformation pipeline.

    Returns:
        np.ndarray: The updated mask after tracking.
    """
    device = "cuda"
    H, W = rgb.shape[:2]

    # Process the image and mask
    rgb_tensor = xmem_im_transform(rgb).unsqueeze(0).cuda()

    if mask is not None:
        mask_tensor = xmem_mask_transform(mask).unsqueeze(0).cuda()
        converted_masks = xmem_mapper.convert_mask(
            mask_tensor[0].cpu().numpy(), exhaustive=True
        )[0]
        converted_masks = converted_masks.to(device)
        xmem_processor.set_all_labels(list(xmem_mapper.remappings.values()))

    prob = xmem_processor.step(
        rgb_tensor[0],
        converted_masks[0] if mask is not None else None,
        list(xmem_mapper.remappings.values()) if mask is not None else None,
        end=False,
    )
    prob = F.interpolate(
        prob.unsqueeze(1),
        (H, W),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

    out_mask = torch.argmax(prob, dim=0)
    out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
    out_mask = xmem_mapper.remap_index_mask(out_mask)

    return out_mask


if __name__ == "__main__":
    # Example usage
    curr_path = os.path.dirname(os.path.abspath(__file__))
    img = cv2.imread(f"{curr_path}/sam_test.jpg")
    predictor = load_sam()
    mask = label_image(img, predictor)
    print("Generated mask shape:", mask.shape)

    # Load XMem model
    xmem_processor, xmem_mapper, xmem_im_transform, xmem_mask_transform = load_XMem()
    # Example tracking
    mask = track_mask(
        img, mask, xmem_processor, xmem_mapper, xmem_im_transform, xmem_mask_transform
    )
    mask_vis = mask / np.max(mask) * 255
    mask_vis = np.repeat(mask_vis[:, :, None], 3, axis=2)
    mask_vis = mask_vis.astype(np.uint8)
    mask_vis = cv2.resize(mask_vis, (img.shape[1] // 4, img.shape[0] // 4))
    cv2.imshow("Tracked Mask", mask_vis)
    cv2.waitKey(30)
    for _ in range(100):
        mask = track_mask(
            img,
            None,
            xmem_processor,
            xmem_mapper,
            xmem_im_transform,
            xmem_mask_transform,
        )
        mask_vis = mask / np.max(mask) * 255
        mask_vis = np.repeat(mask_vis[:, :, None], 3, axis=2)
        mask_vis = mask_vis.astype(np.uint8)
        mask_vis = cv2.resize(mask_vis, (img.shape[1] // 4, img.shape[0] // 4))
        cv2.imshow("Tracked Mask", mask_vis)
        cv2.waitKey(30)
