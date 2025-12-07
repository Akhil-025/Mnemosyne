# app/processing/burst_culling.py

"""
Smart burst detection and culling
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib

from sqlalchemy.orm import Session, joinedload

from app.db.models import File, AnalysisResult
from app.core.face_analysis import FaceAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BurstGroup:
    """Group of burst photos"""
    id: str
    files: List[Path]
    timestamps: List[datetime]
    scores: Dict[Path, float]
    best_file: Optional[Path] = None
    keep_files: List[Path] = None
    
    def __post_init__(self):
        if self.keep_files is None:
            self.keep_files = []


class BurstDetector:
    """Detect burst photo groups"""
    
    def __init__(self, time_threshold: float = 1.0, 
                 filename_pattern: str = "IMG_%Y%m%d_%H%M%S"):
        self.time_threshold = time_threshold  # seconds between shots
        self.filename_pattern = filename_pattern
        
    async def detect_bursts(self, files: List[Path], 
                          timestamps: List[datetime]) -> List[BurstGroup]:
        """Detect burst groups from file list"""
        if len(files) != len(timestamps):
            raise ValueError("Files and timestamps must be same length")
        
        # Sort by timestamp
        sorted_items = sorted(zip(files, timestamps), key=lambda x: x[1])
        files_sorted = [item[0] for item in sorted_items]
        timestamps_sorted = [item[1] for item in sorted_items]
        
        bursts = []
        current_burst = []
        current_times = []
        
        for i, (file, timestamp) in enumerate(zip(files_sorted, timestamps_sorted)):
            if i == 0:
                current_burst.append(file)
                current_times.append(timestamp)
                continue
            
            # Check time difference
            time_diff = (timestamp - timestamps_sorted[i-1]).total_seconds()
            
            if time_diff <= self.time_threshold:
                # Same burst
                current_burst.append(file)
                current_times.append(timestamp)
            else:
                # End of burst
                if len(current_burst) > 1:
                    burst_id = self._generate_burst_id(current_burst, current_times)
                    bursts.append(BurstGroup(
                        id=burst_id,
                        files=current_burst.copy(),
                        timestamps=current_times.copy()
                    ))
                
                # Start new burst
                current_burst = [file]
                current_times = [timestamp]
        
        # Add last burst
        if len(current_burst) > 1:
            burst_id = self._generate_burst_id(current_burst, current_times)
            bursts.append(BurstGroup(
                id=burst_id,
                files=current_burst.copy(),
                timestamps=current_times.copy()
            ))
        
        return bursts
    
    def _generate_burst_id(self, files: List[Path], timestamps: List[datetime]) -> str:
        """Generate unique ID for burst group"""
        # Use first file's name and timestamp
        first_file = files[0]
        first_time = timestamps[0]
        
        # Create hash from file names and timestamps
        hash_input = ''.join([str(f) + str(t) for f, t in zip(files, timestamps)])
        burst_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        return f"burst_{first_time.strftime('%Y%m%d_%H%M%S')}_{burst_hash}"
    
    async def detect_from_directory(self, directory: Path, 
                                  recursive: bool = True) -> List[BurstGroup]:
        """Detect bursts in directory by analyzing file patterns"""
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.heic', '.tiff'}
        files = []
        
        if recursive:
            for ext in image_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in image_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        # Get timestamps from filenames or file system
        files_with_times = []
        for file_path in files:
            timestamp = await self._extract_timestamp(file_path)
            if timestamp:
                files_with_times.append((file_path, timestamp))
        
        # Sort by timestamp
        files_with_times.sort(key=lambda x: x[1])
        
        if not files_with_times:
            return []
        
        # Extract lists
        files_list = [item[0] for item in files_with_times]
        timestamps_list = [item[1] for item in files_with_times]
        
        # Detect bursts
        return await self.detect_bursts(files_list, timestamps_list)
    
    async def _extract_timestamp(self, file_path: Path) -> Optional[datetime]:
        """Extract timestamp from filename or EXIF"""
        # Try to parse from filename first
        try:
            # Common patterns
            patterns = [
                # iOS: IMG_YYYYMMDD_HHMMSS.jpg
                r'IMG_(\d{8})_(\d{6})',
                # Android: YYYYMMDD_HHMMSS.jpg
                r'(\d{8})_(\d{6})',
                # Samsung: YYYYMMDD_HHMMSS.jpg
                r'(\d{14})',
                # Camera: DSC_YYYYMMDD_HHMMSS.jpg
                r'DSC_(\d{8})_(\d{6})',
            ]
            
            import re
            for pattern in patterns:
                match = re.search(pattern, file_path.stem)
                if match:
                    if len(match.groups()) == 2:
                        date_str = match.group(1)
                        time_str = match.group(2)
                        return datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
                    else:
                        # Single group (14 digits)
                        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        except:
            pass
        
        # Fallback to file modification time
        try:
            mtime = file_path.stat().st_mtime
            return datetime.fromtimestamp(mtime)
        except:
            return None


class BurstCulling:
    """Select best photos from burst groups"""
    
    def __init__(self, config: Dict, db_session: Session):
        self.config = config
        self.db = db_session
        self.face_analyzer = FaceAnalyzer()
        
        # Scoring weights
        self.weights = {
            'sharpness': 0.3,
            'faces': 0.25,
            'eye_open': 0.2,
            'aesthetic': 0.15,
            'composition': 0.1
        }
    
    async def score_photos(self, burst_group: BurstGroup) -> Dict[Path, float]:
        """Score photos in burst group"""
        scores = {}
        
        for file_path in burst_group.files:
            score = 0.0
            
            # 1. Sharpness score
            sharpness = await self._calculate_sharpness(file_path)
            score += sharpness * self.weights['sharpness']
            
            # 2. Face detection and quality
            face_score = await self._calculate_face_score(file_path)
            score += face_score * self.weights['faces']
            
            # 3. Eye openness (if faces detected)
            eye_score = await self._calculate_eye_score(file_path)
            score += eye_score * self.weights['eye_open']
            
            # 4. Aesthetic score
            aesthetic = await self._get_aesthetic_score(file_path)
            score += aesthetic * self.weights['aesthetic']
            
            # 5. Composition score (rule of thirds, etc.)
            composition = await self._calculate_composition_score(file_path)
            score += composition * self.weights['composition']
            
            scores[file_path] = score
        
        return scores
    
    async def _calculate_sharpness(self, file_path: Path) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            import cv2
            import numpy as np
            
            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.5
            
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize (typical range 0-1000)
            normalized = min(1.0, variance / 500.0)
            return normalized
            
        except Exception as e:
            logger.error(f"Error calculating sharpness for {file_path}: {e}")
            return 0.5
    
    async def _calculate_face_score(self, file_path: Path) -> float:
        """Calculate score based on face detection"""
        try:
            faces = self.face_analyzer.detect_faces(file_path)
            
            if not faces:
                return 0.3  # No faces penalty
            
            # Score based on number of faces and confidence
            total_confidence = sum(face.confidence for face in faces)
            avg_confidence = total_confidence / len(faces)
            
            # More faces is generally better for group photos
            face_count_score = min(1.0, len(faces) / 10.0)
            
            return (avg_confidence * 0.7) + (face_count_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error calculating face score for {file_path}: {e}")
            return 0.5
    
    async def _calculate_eye_score(self, file_path: Path) -> float:
        """Calculate score based on eye openness"""
        try:
            faces = self.face_analyzer.detect_faces(file_path)
            
            if not faces:
                return 0.5  # Neutral score for no faces
            
            eye_scores = []
            for face in faces:
                if face.eye_open is not None:
                    eye_scores.append(face.eye_open)
            
            if not eye_scores:
                return 0.5
            
            return sum(eye_scores) / len(eye_scores)
            
        except Exception as e:
            logger.error(f"Error calculating eye score for {file_path}: {e}")
            return 0.5
    
    async def _get_aesthetic_score(self, file_path: Path) -> float:
        """Get aesthetic score from database or calculate"""
        # Check database first
        file_record = self.db.query(File).filter(
            File.file_path == str(file_path)
        ).first()
        
        if file_record:
            analysis = self.db.query(AnalysisResult).filter(
                AnalysisResult.file_id == file_record.id
            ).first()
            
            if analysis and analysis.aesthetic_score is not None:
                return analysis.aesthetic_score / 10.0  # Normalize 0-10 to 0-1
        
        # Fallback: calculate simple aesthetic score
        return await self._calculate_simple_aesthetic(file_path)
    
    async def _calculate_simple_aesthetic(self, file_path: Path) -> float:
        """Calculate simple aesthetic score based on image properties"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            
            with Image.open(file_path) as img:
                # Aspect ratio score
                width, height = img.size
                aspect_ratio = width / height
                
                # Preferred ratios: 4:3, 3:2, 16:9, 1:1
                preferred_ratios = [4/3, 3/2, 16/9, 1]
                aspect_score = 1.0 - min(abs(np.log(aspect_ratio / r)) for r in preferred_ratios) / 2.0
                
                # Brightness score (prefer well-exposed)
                if img.mode != 'RGB':
                    img_rgb = img.convert('RGB')
                else:
                    img_rgb = img
                
                np_img = np.array(img_rgb)
                brightness = np.mean(np_img) / 255.0
                brightness_score = 1.0 - abs(brightness - 0.5)  # Peak at 0.5 brightness
                
                # Colorfulness score
                lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                colorfulness = np.std(a) + np.std(b)
                color_score = min(1.0, colorfulness / 100.0)
                
                # Combine scores
                return (aspect_score * 0.3) + (brightness_score * 0.4) + (color_score * 0.3)
                
        except Exception as e:
            logger.error(f"Error calculating aesthetic for {file_path}: {e}")
            return 0.5
    
    async def _calculate_composition_score(self, file_path: Path) -> float:
        """Calculate composition score (rule of thirds, symmetry, etc.)"""
        # Simplified implementation
        # In production, use proper computer vision algorithms
        return 0.5
    
    async def select_best_photos(self, burst_group: BurstGroup, 
                               keep_count: int = 1) -> List[Path]:
        """Select best photos from burst group"""
        # Score all photos
        burst_group.scores = await self.score_photos(burst_group)
        
        # Sort by score
        sorted_photos = sorted(burst_group.scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select best ones
        keep_photos = [photo for photo, score in sorted_photos[:keep_count]]
        
        # Update burst group
        burst_group.best_file = keep_photos[0] if keep_photos else None
        burst_group.keep_files = keep_photos
        
        return keep_photos
    
    async def cull_burst_group(self, burst_group: BurstGroup, 
                             action: str = 'mark') -> Dict[Path, str]:
        """Cull burst group by marking or moving unwanted photos"""
        if not burst_group.keep_files:
            await self.select_best_photos(burst_group)
        
        actions = {}
        
        for file_path in burst_group.files:
            if file_path in burst_group.keep_files:
                actions[file_path] = 'keep'
            else:
                actions[file_path] = action
                
                if action == 'delete':
                    await self._delete_file(file_path)
                elif action == 'move':
                    await self._move_to_trash(file_path)
                elif action == 'mark':
                    await self._mark_as_duplicate(file_path, burst_group.best_file)
        
        return actions
    
    async def _delete_file(self, file_path: Path) -> bool:
        """Delete file permanently"""
        try:
            file_path.unlink()
            logger.info(f"Deleted: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {e}")
            return False
    
    async def _move_to_trash(self, file_path: Path) -> bool:
        """Move file to trash/recycle bin"""
        try:
            import send2trash
            send2trash.send2trash(str(file_path))
            logger.info(f"Moved to trash: {file_path}")
            return True
        except ImportError:
            logger.warning("send2trash not installed, deleting instead")
            return await self._delete_file(file_path)
        except Exception as e:
            logger.error(f"Error moving to trash {file_path}: {e}")
            return False
    
    async def _mark_as_duplicate(self, file_path: Path, original_file: Path) -> bool:
        """Mark file as duplicate in database"""
        try:
            # Find file in database
            file_record = self.db.query(File).filter(
                File.file_path == str(file_path)
            ).first()
            
            if file_record:
                file_record.error = f"Burst duplicate of {original_file}"
                file_record.processed = True
                self.db.commit()
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error marking duplicate {file_path}: {e}")
            return False
    
    async def process_directory(self, directory: Path, 
                              keep_count: int = 1,
                              action: str = 'mark') -> Dict:
        """Process directory for burst culling"""
        detector = BurstDetector()
        bursts = await detector.detect_from_directory(directory)
        
        results = {
            'total_bursts': len(bursts),
            'total_photos': 0,
            'kept_photos': 0,
            'culled_photos': 0,
            'burst_details': []
        }
        
        for burst in bursts:
            # Select best photos
            keep_photos = await self.select_best_photos(burst, keep_count)
            
            # Cull unwanted photos
            actions = await self.cull_burst_group(burst, action)
            
            # Update results
            results['total_photos'] += len(burst.files)
            results['kept_photos'] += len(keep_photos)
            results['culled_photos'] += (len(burst.files) - len(keep_photos))
            
            results['burst_details'].append({
                'id': burst.id,
                'total_files': len(burst.files),
                'kept_files': [str(p) for p in keep_photos],
                'best_file': str(burst.best_file) if burst.best_file else None,
                'actions': {str(k): v for k, v in actions.items()}
            })
        
        return results