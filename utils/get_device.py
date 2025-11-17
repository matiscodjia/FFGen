import logging
import torch

def get_device() -> str:
    """
    Detect and return the best available device for PyTorch.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """
    if torch.cuda.is_available():
        device = 'cuda'
        device_name = torch.cuda.get_device_name(0)
        logger = logging.getLogger(__name__)
        logger.info(f"Using CUDA device: {device_name}")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logger = logging.getLogger(__name__)
        logger.info("Using MPS (Apple Silicon) device")
    else:
        device = 'cpu'
        logger = logging.getLogger(__name__)
        logger.warning("No GPU available, using CPU")

    return device