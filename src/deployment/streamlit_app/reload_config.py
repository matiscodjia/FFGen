"""
Utility to force reload config without restarting Streamlit
"""

import importlib
import sys

def reload_config():
    """Force reload the config module"""
    if 'config' in sys.modules:
        import config
        importlib.reload(config)
        return True
    return False

if __name__ == "__main__":
    reload_config()
    print("Config reloaded!")
