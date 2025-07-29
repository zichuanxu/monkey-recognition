"""File and directory utilities."""

import os
import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from .logging import get_logger

logger = get_logger(__name__)


def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path to create.
    """
    os.makedirs(directory, exist_ok=True)


def safe_remove(path: str) -> bool:
    """Safely remove file or directory.

    Args:
        path: Path to remove.

    Returns:
        True if successful, False otherwise.
    """
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        return True
    except Exception as e:
        logger.error(f"Error removing {path}: {e}")
        return False


def copy_file(src: str, dst: str) -> bool:
    """Copy file from source to destination.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        logger.error(f"Error copying {src} to {dst}: {e}")
        return False


def move_file(src: str, dst: str) -> bool:
    """Move file from source to destination.

    Args:
        src: Source file path.
        dst: Destination file path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        ensure_dir(os.path.dirname(dst))
        shutil.move(src, dst)
        return True
    except Exception as e:
        logger.error(f"Error moving {src} to {dst}: {e}")
        return False


def get_file_size(file_path: str) -> Optional[int]:
    """Get file size in bytes.

    Args:
        file_path: Path to file.

    Returns:
        File size in bytes, or None if file doesn't exist.
    """
    try:
        return os.path.getsize(file_path)
    except:
        return None


def list_files(directory: str, pattern: str = "*", recursive: bool = True) -> List[str]:
    """List files in directory matching pattern.

    Args:
        directory: Directory to search.
        pattern: File pattern to match.
        recursive: Whether to search recursively.

    Returns:
        List of matching file paths.
    """
    try:
        path = Path(directory)
        if recursive:
            return [str(p) for p in path.rglob(pattern) if p.is_file()]
        else:
            return [str(p) for p in path.glob(pattern) if p.is_file()]
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        return []


def list_directories(directory: str, recursive: bool = False) -> List[str]:
    """List subdirectories in directory.

    Args:
        directory: Directory to search.
        recursive: Whether to search recursively.

    Returns:
        List of subdirectory paths.
    """
    try:
        path = Path(directory)
        if recursive:
            return [str(p) for p in path.rglob("*") if p.is_dir()]
        else:
            return [str(p) for p in path.glob("*") if p.is_dir()]
    except Exception as e:
        logger.error(f"Error listing directories in {directory}: {e}")
        return []


def save_json(data: Any, file_path: str, indent: int = 2) -> bool:
    """Save data to JSON file.

    Args:
        data: Data to save.
        file_path: Path to save file.
        indent: JSON indentation.

    Returns:
        True if successful, False otherwise.
    """
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {file_path}: {e}")
        return False


def load_json(file_path: str) -> Optional[Any]:
    """Load data from JSON file.

    Args:
        file_path: Path to JSON file.

    Returns:
        Loaded data, or None if failed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return None


def save_yaml(data: Any, file_path: str) -> bool:
    """Save data to YAML file.

    Args:
        data: Data to save.
        file_path: Path to save file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving YAML to {file_path}: {e}")
        return False


def load_yaml(file_path: str) -> Optional[Any]:
    """Load data from YAML file.

    Args:
        file_path: Path to YAML file.

    Returns:
        Loaded data, or None if failed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading YAML from {file_path}: {e}")
        return None


def save_pickle(data: Any, file_path: str) -> bool:
    """Save data to pickle file.

    Args:
        data: Data to save.
        file_path: Path to save file.

    Returns:
        True if successful, False otherwise.
    """
    try:
        ensure_dir(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Error saving pickle to {file_path}: {e}")
        return False


def load_pickle(file_path: str) -> Optional[Any]:
    """Load data from pickle file.

    Args:
        file_path: Path to pickle file.

    Returns:
        Loaded data, or None if failed.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle from {file_path}: {e}")
        return None


def get_directory_structure(directory: str, max_depth: int = 3) -> Dict[str, Any]:
    """Get directory structure as nested dictionary.

    Args:
        directory: Root directory.
        max_depth: Maximum depth to traverse.

    Returns:
        Nested dictionary representing directory structure.
    """
    def _get_structure(path: str, current_depth: int) -> Dict[str, Any]:
        if current_depth > max_depth:
            return {}

        structure = {}
        try:
            for item in os.listdir(path):
                item_path = os.path.join(path, item)
                if os.path.isdir(item_path):
                    structure[item] = {
                        'type': 'directory',
                        'children': _get_structure(item_path, current_depth + 1)
                    }
                else:
                    structure[item] = {
                        'type': 'file',
                        'size': get_file_size(item_path)
                    }
        except PermissionError:
            pass

        return structure

    return _get_structure(directory, 0)


def find_files_by_extension(directory: str, extensions: Union[str, List[str]]) -> List[str]:
    """Find files by extension(s) in directory.

    Args:
        directory: Directory to search.
        extensions: File extension(s) to search for.

    Returns:
        List of matching file paths.
    """
    if isinstance(extensions, str):
        extensions = [extensions]

    # Ensure extensions start with dot
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

    files = []
    for ext in extensions:
        files.extend(list_files(directory, f"*{ext}"))

    return sorted(files)


def get_relative_path(file_path: str, base_path: str) -> str:
    """Get relative path from base path.

    Args:
        file_path: Full file path.
        base_path: Base path to calculate relative from.

    Returns:
        Relative path.
    """
    return os.path.relpath(file_path, base_path)


def is_empty_directory(directory: str) -> bool:
    """Check if directory is empty.

    Args:
        directory: Directory path to check.

    Returns:
        True if directory is empty, False otherwise.
    """
    try:
        return len(os.listdir(directory)) == 0
    except:
        return False