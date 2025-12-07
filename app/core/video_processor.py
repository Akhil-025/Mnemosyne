# core/video_processor.py

"""
Video processing utilities for extracting keyframes
"""
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Extract keyframes and metadata from videos"""
    
    @staticmethod
    def extract_middle_keyframe(video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Extract middle frame from video for analysis
        Returns path to extracted frame image
        """
        if not video_path.exists():
            logger.error(f"Video file not found: {video_path}")
            return None
        
        try:
            # Get video duration using ffprobe
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip()) if result.stdout else 0
            
            if duration <= 0:
                logger.warning(f"Could not get duration for {video_path}, using first frame")
                return VideoProcessor.extract_frame_at_time(video_path, 0, output_path)
            
            # Extract frame at middle of video
            middle_time = duration / 2
            return VideoProcessor.extract_frame_at_time(video_path, middle_time, output_path)
            
        except Exception as e:
            logger.error(f"Error extracting keyframe from {video_path}: {e}")
            return None
    
    @staticmethod
    def extract_frame_at_time(video_path: Path, timestamp: float, output_path: Optional[Path] = None) -> Optional[Path]:
        """Extract frame at specific timestamp"""
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix='.jpg'))
        
        try:
            cmd = [
                'ffmpeg', '-ss', str(timestamp),
                '-i', str(video_path),
                '-frames:v', '1',
                '-q:v', '2',  # Quality 2-31, lower is better
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and output_path.exists():
                return output_path
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting frame at {timestamp}s from {video_path}: {e}")
            return None
    
    @staticmethod
    def extract_keyframes(video_path: Path, interval_seconds: int = 10, 
                         output_dir: Optional[Path] = None) -> List[Path]:
        """Extract keyframes at regular intervals"""
        if not video_path.exists():
            return []
        
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Get video duration
            cmd = [
                'ffprobe', '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip()) if result.stdout else 0
            
            if duration <= 0:
                return []
            
            # Extract frames at intervals
            frames = []
            for t in range(0, int(duration), interval_seconds):
                frame_path = output_dir / f"frame_{t:04d}.jpg"
                extracted = VideoProcessor.extract_frame_at_time(video_path, t, frame_path)
                if extracted:
                    frames.append(extracted)
            
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting keyframes from {video_path}: {e}")
            return []
    
    @staticmethod
    def get_video_metadata(video_path: Path) -> dict:
        """Extract comprehensive video metadata using ffprobe"""
        import json
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                metadata = json.loads(result.stdout)
                
                # Extract useful information
                info = {
                    'duration': float(metadata.get('format', {}).get('duration', 0)),
                    'size': int(metadata.get('format', {}).get('size', 0)),
                    'bitrate': int(metadata.get('format', {}).get('bit_rate', 0)),
                    'format': metadata.get('format', {}).get('format_name', ''),
                    'creation_time': metadata.get('format', {}).get('tags', {}).get('creation_time'),
                }
                
                # Video stream info
                for stream in metadata.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        info.update({
                            'width': stream.get('width'),
                            'height': stream.get('height'),
                            'codec': stream.get('codec_name'),
                            'fps': eval(stream.get('avg_frame_rate', '0/1')) if '/' in stream.get('avg_frame_rate', '0/1') else 0,
                            'rotation': int(stream.get('tags', {}).get('rotate', 0))
                        })
                        break
                
                return info
            return {}
            
        except Exception as e:
            logger.error(f"Error getting video metadata for {video_path}: {e}")
            return {}
    
    @staticmethod
    def generate_video_thumbnail(video_path: Path, output_path: Path, 
                               width: int = 320, height: int = 180) -> bool:
        """Generate thumbnail for video"""
        try:
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-vf', f'scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2',
                '-frames:v', '1',
                '-q:v', '2',
                '-y',
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and output_path.exists()
            
        except Exception as e:
            logger.error(f"Error generating thumbnail for {video_path}: {e}")
            return False
    
    @staticmethod
    def detect_scene_changes(video_path: Path, threshold: float = 0.3) -> List[float]:
        """
        Detect scene changes in video
        Returns list of timestamps where scene changes occur
        """
        try:
            # Use OpenCV for scene detection
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or frame_count <= 0:
                return []
            
            scene_changes = []
            prev_frame = None
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale and resize for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))
                
                if prev_frame is not None:
                    # Calculate difference between frames
                    diff = cv2.absdiff(gray, prev_frame)
                    diff_mean = np.mean(diff)
                    
                    if diff_mean > threshold * 255:  # Threshold normalized to 0-255
                        timestamp = frame_idx / fps
                        scene_changes.append(timestamp)
                
                prev_frame = gray
                frame_idx += 1
                
                # Skip frames for speed (analyze 1 frame per second)
                skip_frames = int(fps) - 1
                for _ in range(skip_frames):
                    cap.grab()
                    frame_idx += 1
            
            cap.release()
            return scene_changes
            
        except Exception as e:
            logger.error(f"Error detecting scene changes in {video_path}: {e}")
            return []