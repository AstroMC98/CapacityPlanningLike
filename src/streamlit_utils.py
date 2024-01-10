import io
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import toml
from PIL import Image


def get_project_root() -> str:
    """Returns project root path.

    Returns
    -------
    str
        Project root path.
    """
    return str(Path(__file__).parent.parent)

@st.cache_data(ttl=300)
def load_config(
    config_settings_filename: str, config_streamlit_filename: str, config_instructions_filename: str, config_readme_filename: str
):
    """Loads configuration files.

    Parameters
    ----------
    config_streamlit_filename : str
        Filename of lib configuration file.
    config_instructions_filename : str
        Filename of custom config instruction file.
    config_readme_filename : str
        Filename of readme configuration file.

    Returns
    -------
    dict
        Lib configuration file.
    dict
        Readme configuration file.
    """
    config_settings  = toml.load(Path(get_project_root()) / f"config/{config_settings_filename}")
    config_streamlit = toml.load(Path(get_project_root()) / f"config/{config_streamlit_filename}")
    config_instructions = toml.load(
        Path(get_project_root()) / f"config/{config_instructions_filename}"
    )
    config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
    return dict(config_settings), dict(config_streamlit), dict(config_instructions), dict(config_readme)

@st.cache_data(ttl=300)
def load_target_config(target_config_filename : str):
    config_file = toml.load(Path(get_project_root()) / f"config/{target_config_filename}")
    return dict(config_file)

@st.cache_data(ttl=300)
def edit_target_config(new_config, target_config_filename):
    with open(Path(get_project_root()) / f"config/{target_config_filename}", 'w') as f:
        toml.dump(new_config, f)

@st.cache_data(ttl=300)
def load_image(image_name: str) -> Image:
    """Displays an image.

    Parameters
    ----------
    image_name : str
        Local path of the image.

    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(Path(get_project_root()) / f"references/{image_name}")