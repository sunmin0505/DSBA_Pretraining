import os
from omegaconf import OmegaConf, DictConfig


def load_config(config_name: str = 'default') -> DictConfig:
    """
    configs 폴더 내의 .yaml을 읽어서 DictConfig 반환
    """
    cfg_path = os.path.join("configs", f"{config_name}.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    
    return OmegaConf.load(cfg_path)