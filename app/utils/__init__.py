# app/utils/__init__.py

"""
Mnemosyne Utilities
Digital Life Archival System - Utility Functions
"""
from .config import Config, load_config, save_config
from .logger import setup_logging, get_logger
from .file_utils import (
    get_file_type, is_image_file, is_video_file, is_audio_file,
    get_file_size, get_file_hash, get_perceptual_hash,
    create_directory_structure, safe_move_file, safe_copy_file,
    generate_unique_filename, normalize_path, get_mime_type
)
from .geocoding import GeoCoder, reverse_geocode, get_timezone
from .windows_compat import (
    is_windows, get_windows_special_folder, create_windows_junction,
    get_windows_drives, is_hidden_windows, normalize_windows_path,
    get_windows_username
)
from .image_utils import (
    calculate_sharpness, extract_color_palette, calculate_brightness,
    calculate_contrast, resize_image, create_thumbnail, rotate_image,
    get_image_dimensions, get_image_metadata
)

__all__ = [
    'Config', 'load_config', 'save_config',
    'setup_logging', 'get_logger',
    'get_file_type', 'is_image_file', 'is_video_file', 'is_audio_file',
    'get_file_size', 'get_file_hash', 'get_perceptual_hash',
    'create_directory_structure', 'safe_move_file', 'safe_copy_file',
    'generate_unique_filename', 'normalize_path', 'get_mime_type',
    'GeoCoder', 'reverse_geocode', 'get_timezone',
    'is_windows', 'get_windows_special_folder', 'create_windows_junction',
    'get_windows_drives', 'is_hidden_windows', 'normalize_windows_path',
    'get_windows_username',
    'calculate_sharpness', 'extract_color_palette', 'calculate_brightness',
    'calculate_contrast', 'resize_image', 'create_thumbnail', 'rotate_image',
    'get_image_dimensions', 'get_image_metadata',
]