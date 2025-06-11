import os
from pathlib import Path

from groundingdino.util.inference import Model


def load_gdino_model(device: str) -> Model:
    curr_path = os.path.dirname(os.path.abspath(__file__))
    Path(curr_path).mkdir(parents=True, exist_ok=True)
    GDINO_CFG_PATH = f"{curr_path}/ckpts/GroundingDINO_SwinT_OGC.py"
    GDINO_CKPT_PATH = f"{curr_path}/ckpts/groundingdino_swint_ogc.pth"
    if not os.path.exists(GDINO_CFG_PATH):
        os.system(f"wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/config/GroundingDINO_SwinT_OGC.py -P {curr_path}/ckpts/")  # noqa
    assert os.path.exists(GDINO_CFG_PATH), f"GroundingDINO config file not found"
    if not os.path.exists(GDINO_CKPT_PATH):
        os.system(f"wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P {curr_path}/ckpts/")  # noqa
    assert os.path.exists(GDINO_CKPT_PATH), f"GroundingDINO checkpoint file not found"

    grounding_dino_model = Model(
        model_config_path=GDINO_CFG_PATH, 
        model_checkpoint_path=GDINO_CKPT_PATH, 
        device=device
    )
    return grounding_dino_model
