import yaml
from pathlib import Path
from typing import Any, Dict

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        Dictionary containing the configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if config is None:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Validate required fields
        required_fields = ['run_id']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")
