"""
File utilities for Mnemosyne
"""
import os
import hashlib
import shutil
import mimetypes
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from datetime import datetime
import logging
import json
from .image_utils import get_image_dimensions

logger = logging.getLogger(__name__)


# File type detection
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
    '.webp', '.heic', '.heif', '.raw', '.nef', '.cr2', '.arw',
    '.dng', '.orf', '.sr2', '.raf', '.rw2', '.pef', '.srw',
    '.ico', '.svg', '.eps', '.psd', '.ai'
}

VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv',
    '.m4v', '.mpg', '.mpeg', '.3gp', '.mts', '.m2ts', '.ts',
    '.vob', '.ogv', '.rm', '.rmvb', '.asf', '.f4v'
}

AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma',
    '.opus', '.ape', '.alac', '.aiff', '.mid', '.midi', '.amr'
}

DOCUMENT_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.pages',
    '.xls', '.xlsx', '.csv', '.ods', '.numbers', '.ppt', '.pptx',
    '.odp', '.key', '.epub', '.mobi', '.azw', '.azw3'
}

ARCHIVE_EXTENSIONS = {
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz',
    '.tgz', '.tbz2', '.lz', '.lzma', '.z', '.lzh'
}


def get_file_type(file_path: Union[str, Path]) -> str:
    """
    Determine file type from extension
    
    Args:
        file_path: Path to file
        
    Returns:
        File type: 'image', 'video', 'audio', 'document', 'archive', or 'other'
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext in IMAGE_EXTENSIONS:
        return 'image'
    elif ext in VIDEO_EXTENSIONS:
        return 'video'
    elif ext in AUDIO_EXTENSIONS:
        return 'audio'
    elif ext in DOCUMENT_EXTENSIONS:
        return 'document'
    elif ext in ARCHIVE_EXTENSIONS:
        return 'archive'
    else:
        return 'other'


def is_image_file(file_path: Union[str, Path]) -> bool:
    """Check if file is an image"""
    return get_file_type(file_path) == 'image'


def is_video_file(file_path: Union[str, Path]) -> bool:
    """Check if file is a video"""
    return get_file_type(file_path) == 'video'


def is_audio_file(file_path: Union[str, Path]) -> bool:
    """Check if file is an audio file"""
    return get_file_type(file_path) == 'audio'


def is_document_file(file_path: Union[str, Path]) -> bool:
    """Check if file is a document"""
    return get_file_type(file_path) == 'document'


def is_archive_file(file_path: Union[str, Path]) -> bool:
    """Check if file is an archive"""
    return get_file_type(file_path) == 'archive'


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    Get file size in bytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in bytes, or 0 if file doesn't exist
    """
    path = Path(file_path)
    try:
        return path.stat().st_size
    except (OSError, FileNotFoundError):
        return 0


def get_file_hash(file_path: Union[str, Path], 
                 algorithm: str = 'sha256',
                 chunk_size: int = 65536) -> Optional[str]:
    """
    Compute hash of file
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)
        chunk_size: Size of chunks to read
        
    Returns:
        Hex digest of hash, or None if error
    """
    path = Path(file_path)
    
    if not path.exists():
        logger.error(f"File not found: {path}")
        return None
    
    try:
        hash_func = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    except Exception as e:
        logger.error(f"Error computing hash for {path}: {e}")
        return None


def get_perceptual_hash(file_path: Union[str, Path]) -> Optional[str]:
    """
    Compute perceptual hash for image similarity
    
    Args:
        file_path: Path to image file
        
    Returns:
        Perceptual hash string, or None if error
    """
    if not is_image_file(file_path):
        return None
    
    try:
        import imagehash
        from PIL import Image
        
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            # Compute multiple perceptual hashes
            ahash = str(imagehash.average_hash(img))
            phash = str(imagehash.phash(img))
            dhash = str(imagehash.dhash(img))
            whash = str(imagehash.whash(img))
            
            # Combine hashes
            return f"{ahash}:{phash}:{dhash}:{whash}"
    
    except ImportError:
        logger.warning("imagehash not installed, perceptual hashing disabled")
        return None
    except Exception as e:
        logger.error(f"Error computing perceptual hash for {file_path}: {e}")
        return None


def get_mime_type(file_path: Union[str, Path]) -> str:
    """
    Guess MIME type from file extension
    
    Args:
        file_path: Path to file
        
    Returns:
        MIME type string
    """
    path = Path(file_path)
    
    # Try python-magic if available
    try:
        import magic
        mime = magic.Magic(mime=True)
        return mime.from_file(str(path))
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"python-magic error: {e}")
    
    # Fallback to mimetypes
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or 'application/octet-stream'


def create_directory_structure(base_dir: Union[str, Path], 
                              structure: List[str]) -> List[Path]:
    """
    Create directory structure
    
    Args:
        base_dir: Base directory
        structure: List of directory names to create
        
    Returns:
        List of created directory paths
    """
    base_path = Path(base_dir)
    created = []
    
    for dir_name in structure:
        dir_path = base_path / dir_name
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            created.append(dir_path)
        except Exception as e:
            logger.error(f"Error creating directory {dir_path}: {e}")
    
    return created


def safe_move_file(source: Union[str, Path], 
                  target: Union[str, Path],
                  overwrite: bool = False) -> bool:
    """
    Safely move file with error handling
    
    Args:
        source: Source file path
        target: Target file path
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    source_path = Path(source)
    target_path = Path(target)
    
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return False
    
    # Check if target exists
    if target_path.exists():
        if not overwrite:
            logger.error(f"Target file already exists: {target_path}")
            return False
        
        # Backup existing file
        backup_path = target_path.with_suffix(f"{target_path.suffix}.bak")
        try:
            shutil.move(target_path, backup_path)
            logger.info(f"Backed up existing file to: {backup_path}")
        except Exception as e:
            logger.error(f"Error backing up existing file: {e}")
            return False
    
    # Create parent directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Use shutil.move which handles cross-device moves
        shutil.move(str(source_path), str(target_path))
        logger.info(f"Moved file: {source_path} -> {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error moving file {source_path} to {target_path}: {e}")
        return False


def safe_copy_file(source: Union[str, Path],
                  target: Union[str, Path],
                  preserve_metadata: bool = True,
                  overwrite: bool = False) -> bool:
    """
    Safely copy file with error handling
    
    Args:
        source: Source file path
        target: Target file path
        preserve_metadata: Whether to preserve file metadata
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful, False otherwise
    """
    source_path = Path(source)
    target_path = Path(target)
    
    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return False
    
    # Check if target exists
    if target_path.exists() and not overwrite:
        logger.error(f"Target file already exists: {target_path}")
        return False
    
    # Create parent directory if it doesn't exist
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if preserve_metadata:
            shutil.copy2(str(source_path), str(target_path))
        else:
            shutil.copy(str(source_path), str(target_path))
        
        logger.info(f"Copied file: {source_path} -> {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error copying file {source_path} to {target_path}: {e}")
        return False


def generate_unique_filename(base_path: Union[str, Path], 
                           suffix: str = "") -> Path:
    """
    Generate unique filename by adding counter if file exists
    
    Args:
        base_path: Base file path
        suffix: Optional suffix to add before counter
        
    Returns:
        Unique file path
    """
    path = Path(base_path)
    
    if not path.exists():
        return path
    
    # Try with suffix first
    if suffix:
        candidate = path.with_stem(f"{path.stem}_{suffix}")
        if not candidate.exists():
            return candidate
    
    # Try with counter
    counter = 1
    while True:
        candidate = path.with_stem(f"{path.stem}_{counter}")
        if not candidate.exists():
            return candidate
        counter += 1


def normalize_path(path: Union[str, Path], 
                  platform: str = None) -> Path:
    """
    Normalize path for cross-platform compatibility
    
    Args:
        path: Path to normalize
        platform: Target platform (windows, linux, macos), auto-detected if None
        
    Returns:
        Normalized Path object
    """
    if platform is None:
        import sys
        platform = 'windows' if sys.platform == 'win32' else 'posix'
    
    path_str = str(path)
    
    if platform == 'windows':
        # Convert to Windows path
        path_str = path_str.replace('/', '\\')
        # Remove leading slash if present
        if path_str.startswith('\\') and not path_str.startswith('\\\\'):
            path_str = path_str[1:]
    else:
        # Convert to POSIX path
        path_str = path_str.replace('\\', '/')
        # Ensure proper root
        if path_str[1:3] == ':\\':  # Windows drive letter
            path_str = path_str[2:]  # Remove drive letter
    
    return Path(path_str)


def find_files(directory: Union[str, Path],
               patterns: List[str] = None,
               recursive: bool = True,
               min_size: int = 0,
               max_size: int = None) -> List[Path]:
    """
    Find files matching patterns
    
    Args:
        directory: Directory to search
        patterns: List of file patterns (e.g., ['*.jpg', '*.png'])
        recursive: Whether to search recursively
        min_size: Minimum file size in bytes
        max_size: Maximum file size in bytes
        
    Returns:
        List of matching file paths
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.error(f"Directory not found: {dir_path}")
        return []
    
    if patterns is None:
        patterns = ['*']  # Match all files
    
    files = []
    
    for pattern in patterns:
        if recursive:
            found = list(dir_path.rglob(pattern))
        else:
            found = list(dir_path.glob(pattern))
        
        files.extend(found)
    
    # Filter by size
    filtered_files = []
    for file_path in files:
        if file_path.is_file():
            size = file_path.stat().st_size
            
            if size < min_size:
                continue
            
            if max_size is not None and size > max_size:
                continue
            
            filtered_files.append(file_path)
    
    return filtered_files


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive file metadata
    
    Args:
        file_path: Path to file
        
    Returns:
        Dictionary with file metadata
    """
    path = Path(file_path)
    
    if not path.exists():
        return {}
    
    try:
        stat = path.stat()
        
        metadata = {
            'path': str(path),
            'name': path.name,
            'stem': path.stem,
            'suffix': path.suffix,
            'parent': str(path.parent),
            'size': stat.st_size,
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'type': get_file_type(path),
            'mime_type': get_mime_type(path),
            'hash': get_file_hash(path),
            'perceptual_hash': get_perceptual_hash(path) if is_image_file(path) else None,
        }
        
        # Add image-specific metadata
        if is_image_file(path):
            metadata.update(get_image_dimensions(path))
        
        return metadata
    
    except Exception as e:
        logger.error(f"Error getting metadata for {path}: {e}")
        return {}


def compare_files(file1: Union[str, Path], 
                 file2: Union[str, Path],
                 compare_hash: bool = True,
                 compare_content: bool = False) -> Dict[str, Any]:
    """
    Compare two files
    
    Args:
        file1: First file path
        file2: Second file path
        compare_hash: Whether to compare file hashes
        compare_content: Whether to compare file content (slower)
        
    Returns:
        Dictionary with comparison results
    """
    path1 = Path(file1)
    path2 = Path(file2)
    
    result = {
        'identical': False,
        'same_size': False,
        'same_hash': False,
        'same_content': False,
        'file1_exists': path1.exists(),
        'file2_exists': path2.exists(),
    }
    
    if not (result['file1_exists'] and result['file2_exists']):
        return result
    
    # Compare size
    size1 = path1.stat().st_size
    size2 = path2.stat().st_size
    result['same_size'] = (size1 == size2)
    
    if not result['same_size']:
        return result
    
    # Compare hash
    if compare_hash:
        hash1 = get_file_hash(path1)
        hash2 = get_file_hash(path2)
        result['same_hash'] = (hash1 == hash2 and hash1 is not None)
    
    # Compare content
    if compare_content and result['same_hash']:
        try:
            with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
                chunk_size = 65536
                while True:
                    chunk1 = f1.read(chunk_size)
                    chunk2 = f2.read(chunk_size)
                    if chunk1 != chunk2:
                        result['same_content'] = False
                        break
                    if not chunk1:  # End of file
                        result['same_content'] = True
                        break
        except Exception as e:
            logger.error(f"Error comparing file content: {e}")
            result['same_content'] = False
    
    result['identical'] = result['same_size'] and result['same_hash']
    
    return result


def create_temp_file(suffix: str = ".tmp",
                    prefix: str = "mnemosyne_",
                    content: bytes = None) -> Path:
    """
    Create temporary file
    
    Args:
        suffix: File suffix
        prefix: File prefix
        content: Optional content to write
        
    Returns:
        Path to temporary file
    """
    temp_dir = tempfile.gettempdir()
    temp_file = tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix=prefix,
        dir=temp_dir,
        delete=False
    )
    temp_path = Path(temp_file.name)
    
    if content:
        temp_file.write(content)
    
    temp_file.close()
    
    return temp_path


def cleanup_temp_files(pattern: str = "mnemosyne_*",
                      max_age_hours: int = 24):
    """
    Clean up old temporary files
    
    Args:
        pattern: File pattern to match
        max_age_hours: Maximum age in hours
    """
    temp_dir = Path(tempfile.gettempdir())
    now = datetime.now()
    
    for temp_file in temp_dir.glob(pattern):
        try:
            mtime = datetime.fromtimestamp(temp_file.stat().st_mtime)
            age_hours = (now - mtime).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                temp_file.unlink()
                logger.debug(f"Cleaned up temp file: {temp_file}")
        except Exception as e:
            logger.debug(f"Error cleaning up temp file {temp_file}: {e}")