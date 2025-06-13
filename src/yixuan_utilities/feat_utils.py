import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


def extract_dinov2_features(
    dinov2_model: torch.nn.Module,
    imgs: np.ndarray,
    patch_h: int,
    patch_w: int,
    device: str,
) -> torch.Tensor:
    assert imgs.ndim == 4, "imgs must be a 4D array"
    assert imgs.max() > 1 and imgs.max() < 256, "imgs must be in [0, 255]"
    assert imgs.min() >= 0, "imgs must be in [0, 255]"
    assert imgs.dtype == np.uint8, "imgs must be uint8"

    K, H, W, _ = imgs.shape

    feat_dim = 1024  # vitl14

    transform = T.Compose(
        [
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=device)
    for j in range(K):
        img = Image.fromarray(imgs[j])
        imgs_tensor[j] = transform(img)[:3]
    with torch.no_grad():
        features_dict = dinov2_model.forward_features(imgs_tensor)
        features = features_dict["x_norm_patchtokens"]
        features = features.reshape((K, patch_h, patch_w, feat_dim))
    return features
