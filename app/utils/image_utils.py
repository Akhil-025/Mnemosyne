"""
Image utilities for Mnemosyne
"""
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
from PIL import Image, ImageOps, ExifTags
import io

logger = logging.getLogger(__name__)


def calculate_sharpness(image_path: Union[str, Path]) -> float:
    """
    Calculate image sharpness using Laplacian variance
    
    Args:
        image_path: Path to image file
        
    Returns:
        Sharpness score (0.0 to 1.0)
    """
    try:
        import cv2
        
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"Failed to read image: {image_path}")
            return 0.0
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize (typical range 0-1000 for images)
        normalized = min(1.0, variance / 500.0)
        return normalized
        
    except ImportError:
        logger.warning("OpenCV not installed, sharpness calculation disabled")
        return 0.5
    except Exception as e:
        logger.error(f"Error calculating sharpness for {image_path}: {e}")
        return 0.0


def extract_color_palette(image_path: Union[str, Path], 
                         n_colors: int = 5) -> List[List[int]]:
    """
    Extract dominant colors from image
    
    Args:
        image_path: Path to image file
        n_colors: Number of dominant colors to extract
        
    Returns:
        List of RGB colors (each as [R, G, B])
    """
    try:
        from sklearn.cluster import KMeans
        
        with Image.open(image_path) as img:
            # Resize for performance
            img = img.resize((100, 100))
            
            # Convert to numpy array
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # Skip if too few pixels
            if len(pixels) < n_colors:
                return []
            
            # Use K-means to find dominant colors
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers (colors)
            colors = kmeans.cluster_centers_.astype(int)
            
            # Sort by cluster size (frequency)
            labels = kmeans.labels_
            cluster_sizes = np.bincount(labels)
            sorted_indices = np.argsort(cluster_sizes)[::-1]
            
            dominant_colors = colors[sorted_indices].tolist()
            return dominant_colors
            
    except ImportError:
        logger.warning("scikit-learn not installed, color palette extraction disabled")
        return []
    except Exception as e:
        logger.error(f"Error extracting color palette from {image_path}: {e}")
        return []


def calculate_brightness(image_path: Union[str, Path]) -> float:
    """
    Calculate average brightness of image
    
    Args:
        image_path: Path to image file
        
    Returns:
        Brightness score (0.0 to 1.0)
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale if necessary
            if img.mode != 'L':
                img = img.convert('L')
            
            # Calculate average pixel value
            pixels = np.array(img)
            brightness = np.mean(pixels) / 255.0
            
            return brightness
            
    except Exception as e:
        logger.error(f"Error calculating brightness for {image_path}: {e}")
        return 0.5


def calculate_contrast(image_path: Union[str, Path]) -> float:
    """
    Calculate image contrast
    
    Args:
        image_path: Path to image file
        
    Returns:
        Contrast score (0.0 to 1.0)
    """
    try:
        with Image.open(image_path) as img:
            # Convert to grayscale if necessary
            if img.mode != 'L':
                img = img.convert('L')
            
            # Calculate standard deviation of pixel values
            pixels = np.array(img)
            contrast = np.std(pixels) / 255.0
            
            return contrast
            
    except Exception as e:
        logger.error(f"Error calculating contrast for {image_path}: {e}")
        return 0.5


def resize_image(image_path: Union[str, Path],
                output_path: Union[str, Path],
                max_size: Tuple[int, int] = (1920, 1080),
                quality: int = 85,
                preserve_aspect_ratio: bool = True) -> bool:
    """
    Resize image to specified dimensions
    
    Args:
        image_path: Input image path
        output_path: Output image path
        max_size: Maximum (width, height)
        quality: JPEG quality (1-100)
        preserve_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Calculate new dimensions
            if preserve_aspect_ratio:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            else:
                img = img.resize(max_size, Image.Resampling.LANCZOS)
            
            # Save resized image
            img.save(output_path, quality=quality, optimize=True)
            
            logger.debug(f"Resized image: {image_path} -> {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        return False


def create_thumbnail(image_path: Union[str, Path],
                    output_path: Union[str, Path],
                    size: Tuple[int, int] = (320, 240),
                    crop: bool = False) -> bool:
    """
    Create thumbnail image
    
    Args:
        image_path: Input image path
        output_path: Output thumbnail path
        size: Thumbnail size (width, height)
        crop: Whether to crop to exact size
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            if crop:
                # Crop to center square then resize
                width, height = img.size
                min_dim = min(width, height)
                
                left = (width - min_dim) // 2
                top = (height - min_dim) // 2
                right = left + min_dim
                bottom = top + min_dim
                
                img = img.crop((left, top, right, bottom))
                img.thumbnail(size, Image.Resampling.LANCZOS)
            else:
                # Preserve aspect ratio
                img.thumbnail(size, Image.Resampling.LANCZOS)
            
            # Save thumbnail
            img.save(output_path, quality=85, optimize=True)
            
            logger.debug(f"Created thumbnail: {image_path} -> {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error creating thumbnail for {image_path}: {e}")
        return False


def rotate_image(image_path: Union[str, Path],
                output_path: Union[str, Path],
                degrees: float) -> bool:
    """
    Rotate image
    
    Args:
        image_path: Input image path
        output_path: Output image path
        degrees: Rotation angle in degrees
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Rotate image
            rotated = img.rotate(degrees, expand=True)
            
            # Save rotated image
            rotated.save(output_path, quality=95)
            
            logger.debug(f"Rotated image {degrees}°: {image_path} -> {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error rotating image {image_path}: {e}")
        return False


def get_image_dimensions(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get image dimensions and aspect ratio
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with dimension information
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            
            return {
                'width': width,
                'height': height,
                'aspect_ratio': width / height,
                'orientation': 'landscape' if width > height else 
                              'portrait' if height > width else 'square',
                'megapixels': (width * height) / 1_000_000,
            }
            
    except Exception as e:
        logger.error(f"Error getting dimensions for {image_path}: {e}")
        return {}


def get_image_metadata(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract image metadata (EXIF)
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image metadata
    """
    try:
        with Image.open(image_path) as img:
            metadata = {
                'format': img.format,
                'mode': img.mode,
                'size': img.size,
                'info': img.info,
            }
            
            # Extract EXIF data
            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif():
                for tag_id, value in img._getexif().items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            
            if exif_data:
                metadata['exif'] = exif_data
                
                # Extract GPS data
                if 'GPSInfo' in exif_data:
                    gps_info = {}
                    for key, val in exif_data['GPSInfo'].items():
                        if key in ExifTags.GPSTAGS:
                            gps_info[ExifTags.GPSTAGS[key]] = val
                    metadata['gps'] = gps_info
            
            return metadata
            
    except Exception as e:
        logger.error(f"Error extracting metadata from {image_path}: {e}")
        return {}


def compress_image(image_path: Union[str, Path],
                  output_path: Union[str, Path],
                  quality: int = 85,
                  optimize: bool = True,
                  progressive: bool = False) -> bool:
    """
    Compress image to reduce file size
    
    Args:
        image_path: Input image path
        output_path: Output image path
        quality: JPEG quality (1-100)
        optimize: Enable optimization
        progressive: Create progressive JPEG
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save with compression
            img.save(
                output_path,
                quality=quality,
                optimize=optimize,
                progressive=progressive
            )
            
            # Compare file sizes
            input_size = Path(image_path).stat().st_size
            output_size = Path(output_path).stat().st_size
            reduction = (1 - output_size / input_size) * 100
            
            logger.debug(f"Compressed image: {reduction:.1f}% reduction, {image_path} -> {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error compressing image {image_path}: {e}")
        return False


def convert_image_format(image_path: Union[str, Path],
                        output_path: Union[str, Path],
                        output_format: str = 'JPEG',
                        quality: int = 95) -> bool:
    """
    Convert image to different format
    
    Args:
        image_path: Input image path
        output_path: Output image path
        output_format: Output format (JPEG, PNG, WEBP, etc.)
        quality: Quality for lossy formats
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(image_path) as img:
            # Handle transparency for formats that don't support it
            if output_format.upper() == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            # Save in new format
            img.save(output_path, format=output_format, quality=quality)
            
            logger.debug(f"Converted image to {output_format}: {image_path} -> {output_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error converting image {image_path}: {e}")
        return False


def detect_blur(image_path: Union[str, Path], 
                threshold: float = 100.0) -> Tuple[bool, float]:
    """
    Detect if image is blurry
    
    Args:
        image_path: Path to image file
        threshold: Blur threshold (higher = less sensitive)
        
    Returns:
        Tuple of (is_blurry, blur_score)
    """
    try:
        import cv2
        
        # Read image in grayscale
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True, 0.0
        
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()
        
        is_blurry = variance < threshold
        blur_score = variance
        
        return is_blurry, blur_score
        
    except ImportError:
        logger.warning("OpenCV not installed, blur detection disabled")
        return False, 0.0
    except Exception as e:
        logger.error(f"Error detecting blur for {image_path}: {e}")
        return False, 0.0


def extract_image_features(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract comprehensive image features
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dictionary with image features
    """
    features = {}
    
    try:
        # Basic dimensions
        dimensions = get_image_dimensions(image_path)
        features.update(dimensions)
        
        # Color features
        features['brightness'] = calculate_brightness(image_path)
        features['contrast'] = calculate_contrast(image_path)
        
        # Color palette
        palette = extract_color_palette(image_path, n_colors=3)
        if palette:
            features['dominant_colors'] = palette
            
            # Calculate average hue
            import colorsys
            hues = []
            for color in palette:
                r, g, b = [c/255 for c in color]
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                hues.append(h * 360)
            
            if hues:
                features['average_hue'] = sum(hues) / len(hues)
        
        # Sharpness
        features['sharpness'] = calculate_sharpness(image_path)
        
        # Blur detection
        is_blurry, blur_score = detect_blur(image_path)
        features['is_blurry'] = is_blurry
        features['blur_score'] = blur_score
        
        # Metadata
        metadata = get_image_metadata(image_path)
        features['metadata'] = metadata
        
        logger.debug(f"Extracted features for {image_path}")
        
    except Exception as e:
        logger.error(f"Error extracting features from {image_path}: {e}")
    
    return features


def create_image_mosaic(image_paths: List[Union[str, Path]],
                       output_path: Union[str, Path],
                       grid_size: Tuple[int, int] = (3, 3),
                       thumbnail_size: Tuple[int, int] = (200, 200)) -> bool:
    """
    Create image mosaic from multiple images
    
    Args:
        image_paths: List of input image paths
        output_path: Output mosaic image path
        grid_size: Grid dimensions (columns, rows)
        thumbnail_size: Size of each thumbnail
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cols, rows = grid_size
        max_images = cols * rows
        
        # Limit number of images
        image_paths = image_paths[:max_images]
        
        # Create blank canvas
        canvas_width = cols * thumbnail_size[0]
        canvas_height = rows * thumbnail_size[1]
        canvas = Image.new('RGB', (canvas_width, canvas_height), (240, 240, 240))
        
        # Place thumbnails on canvas
        for i, img_path in enumerate(image_paths):
            row = i // cols
            col = i % cols
            
            try:
                with Image.open(img_path) as img:
                    # Create thumbnail
                    img.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                    
                    # Calculate position
                    x = col * thumbnail_size[0] + (thumbnail_size[0] - img.width) // 2
                    y = row * thumbnail_size[1] + (thumbnail_size[1] - img.height) // 2
                    
                    # Paste thumbnail
                    canvas.paste(img, (x, y))
            except Exception as e:
                logger.debug(f"Error processing image for mosaic: {img_path}, {e}")
                continue
        
        # Save mosaic
        canvas.save(output_path, quality=90)
        
        logger.info(f"Created mosaic with {len(image_paths)} images: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating image mosaic: {e}")
        return False


def validate_image(image_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Validate image file integrity
    
    Args:
        image_path: Path to image file
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            # Try to load the image data
            img.verify()
            
            # Check file size
            file_size = Path(image_path).stat().st_size
            if file_size == 0:
                return False, "File is empty"
            
            # Check dimensions
            if img.width == 0 or img.height == 0:
                return False, "Invalid image dimensions"
            
            return True, "Image is valid"
            
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def batch_process_images(image_paths: List[Union[str, Path]],
                        operation: str,
                        output_dir: Union[str, Path],
                        **kwargs) -> Dict[str, Any]:
    """
    Batch process multiple images
    
    Args:
        image_paths: List of input image paths
        operation: Operation to perform ('resize', 'compress', 'thumbnail', 'convert')
        output_dir: Output directory
        **kwargs: Operation-specific arguments
        
    Returns:
        Dictionary with processing results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'total': len(image_paths),
        'success': 0,
        'failed': 0,
        'errors': []
    }
    
    for img_path in image_paths:
        img_path = Path(img_path)
        
        try:
            # Generate output filename
            output_name = f"{img_path.stem}_processed{img_path.suffix}"
            output_path = output_dir / output_name
            
            # Perform operation
            success = False
            
            if operation == 'resize':
                success = resize_image(img_path, output_path, **kwargs)
            elif operation == 'compress':
                success = compress_image(img_path, output_path, **kwargs)
            elif operation == 'thumbnail':
                success = create_thumbnail(img_path, output_path, **kwargs)
            elif operation == 'convert':
                success = convert_image_format(img_path, output_path, **kwargs)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                results['errors'].append(f"Failed to process {img_path.name}")
                
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Error processing {img_path.name}: {str(e)}")
    
    return results