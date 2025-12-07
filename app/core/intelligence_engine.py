# core/intelligence_engine.py

"""
Main intelligence engine orchestrating all AI analysis
"""
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from .ollama_client import OllamaClient, ImageAnalysisResult
from .face_analysis import FaceAnalyzer, FaceDetection, FaceDatabase
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


@dataclass
class MediaAnalysis:
    """Complete analysis result for a media file"""
    file_path: Path
    file_hash: str
    file_size: int
    media_type: str  # 'image', 'video', 'audio'
    
    # Image analysis
    image_analysis: Optional[ImageAnalysisResult] = None
    face_detections: List[FaceDetection] = None
    face_clusters: List[int] = None  # Cluster IDs for each face
    
    # Video analysis
    video_metadata: Optional[Dict] = None
    scene_changes: List[float] = None
    
    # Technical metadata
    technical_metadata: Optional[Dict] = None
    exif_data: Optional[Dict] = None
    gps_data: Optional[Dict] = None
    
    # Aesthetic scores
    aesthetic_score: float = 0.0
    sharpness: float = 0.0
    color_palette: List[List[int]] = None
    
    # Processing info
    processing_time: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        # Convert Path objects to strings
        result['file_path'] = str(self.file_path)
        
        # Handle FaceDetection objects
        if self.face_detections:
            result['face_detections'] = [
                {
                    'bbox': det.bbox,
                    'confidence': det.confidence,
                    'gender': det.gender,
                    'age': det.age,
                    'eye_open': det.eye_open
                }
                for det in self.face_detections
            ]
        
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, default=str)


class IntelligenceEngine:
    """Orchestrates all AI analysis for media files"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.ollama = OllamaClient()
        self.face_analyzer = FaceAnalyzer()
        self.face_db = FaceDatabase(Path('./data/faces.db'))
        self.video_processor = VideoProcessor()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'images_analyzed': 0,
            'videos_analyzed': 0,
            'faces_detected': 0,
            'errors': 0
        }
    
    async def analyze_media(self, file_path: Path) -> MediaAnalysis:
        """
        Analyze a media file (image or video)
        Returns complete analysis including faces, caption, tags, etc.
        """
        start_time = datetime.now()
        analysis = MediaAnalysis(
            file_path=file_path,
            file_hash=self._compute_file_hash(file_path),
            file_size=file_path.stat().st_size,
            media_type=self._get_media_type(file_path),
            face_detections=[],
            face_clusters=[],
            scene_changes=[],
            color_palette=[]
        )
        
        try:
            if analysis.media_type == 'image':
                await self._analyze_image(file_path, analysis)
            elif analysis.media_type == 'video':
                await self._analyze_video(file_path, analysis)
            else:
                logger.warning(f"Unsupported media type for {file_path}")
                analysis.error = "Unsupported media type"
            
            # Update statistics
            self.stats['total_processed'] += 1
            if analysis.media_type == 'image':
                self.stats['images_analyzed'] += 1
            elif analysis.media_type == 'video':
                self.stats['videos_analyzed'] += 1
            self.stats['faces_detected'] += len(analysis.face_detections)
            
        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            analysis.error = str(e)
            self.stats['errors'] += 1
        
        analysis.processing_time = (datetime.now() - start_time).total_seconds()
        return analysis
    
    async def _analyze_image(self, file_path: Path, analysis: MediaAnalysis):
        """Analyze image file"""
        # 1. Extract technical metadata
        analysis.technical_metadata = self._extract_image_metadata(file_path)
        analysis.exif_data = analysis.technical_metadata.get('exif', {})
        analysis.gps_data = analysis.technical_metadata.get('gps', {})
        
        # 2. Run Ollama image analysis
        analysis.image_analysis = await self.ollama.analyze_image(file_path)
        
        # 3. Face detection and analysis
        faces = self.face_analyzer.detect_faces(file_path)
        analysis.face_detections = faces
        
        if faces:
            # Extract embeddings for clustering
            embeddings = [face.embedding for face in faces]
            
            # Cluster faces
            if len(embeddings) > 1:
                labels = self.face_analyzer.cluster_faces(embeddings)
                analysis.face_clusters = labels.tolist()
                
                # Add unknown clusters to database
                unique_clusters = set(labels)
                for cluster_id in unique_clusters:
                    if cluster_id >= 0:  # Skip noise
                        cluster_embeddings = [emb for emb, lbl in zip(embeddings, labels) if lbl == cluster_id]
                        db_cluster_id = self.face_db.add_unknown_cluster(cluster_embeddings)
                        # Map local cluster ID to database cluster ID
                        # (This would need more sophisticated mapping in production)
        
        # 4. Calculate aesthetic scores
        analysis.aesthetic_score = await self._calculate_aesthetic_score(file_path)
        analysis.sharpness = await self._calculate_sharpness(file_path)
        analysis.color_palette = await self._extract_color_palette(file_path)
    
    async def _analyze_video(self, file_path: Path, analysis: MediaAnalysis):
        """Analyze video file"""
        # 1. Extract video metadata
        analysis.video_metadata = self.video_processor.get_video_metadata(file_path)
        
        # 2. Extract middle keyframe for analysis
        keyframe_path = self.video_processor.extract_middle_keyframe(file_path)
        
        if keyframe_path:
            try:
                # 3. Analyze keyframe with Ollama
                analysis.image_analysis = await self.ollama.analyze_image(keyframe_path)
                
                # 4. Face detection on keyframe
                faces = self.face_analyzer.detect_faces(keyframe_path)
                analysis.face_detections = faces
                
                # 5. Detect scene changes
                analysis.scene_changes = self.video_processor.detect_scene_changes(file_path)
                
                # 6. Clean up temporary keyframe
                keyframe_path.unlink()
                
            except Exception as e:
                logger.error(f"Error analyzing video keyframe: {e}")
        
        # 7. Generate thumbnail
        thumbnail_path = Path('./data/thumbnails') / f"{file_path.stem}_thumb.jpg"
        thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
        self.video_processor.generate_video_thumbnail(file_path, thumbnail_path)
    
    async def _calculate_aesthetic_score(self, image_path: Path) -> float:
        """Calculate aesthetic score for image"""
        import cv2
        import numpy as np
        
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return 0.5
            
            # Simple heuristic scoring
            score = 0.5
            
            # 1. Rule of thirds compliance
            height, width = img.shape[:2]
            
            # 2. Colorfulness
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            colorfulness = np.std(a) + np.std(b)
            score += min(0.3, colorfulness / 100.0)
            
            # 3. Contrast
            contrast = img.std()
            score += min(0.2, contrast / 100.0)
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating aesthetic score: {e}")
            return 0.5
    
    async def _calculate_sharpness(self, image_path: Path) -> float:
        """Calculate image sharpness using Laplacian variance"""
        import cv2
        import numpy as np
        
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                return 0.0
            
            laplacian = cv2.Laplacian(img, cv2.CV_64F)
            sharpness = laplacian.var()
            
            # Normalize (values typically 0-1000 for images)
            normalized = min(1.0, sharpness / 500.0)
            return normalized
            
        except Exception as e:
            logger.error(f"Error calculating sharpness: {e}")
            return 0.0
    
    async def _extract_color_palette(self, image_path: Path) -> List[List[int]]:
        """Extract dominant colors from image"""
        from PIL import Image
        import numpy as np
        from sklearn.cluster import KMeans
        
        try:
            with Image.open(image_path) as img:
                # Resize for performance
                img = img.resize((100, 100))
                
                # Convert to numpy array
                img_array = np.array(img)
                pixels = img_array.reshape(-1, 3)
                
                # Get dominant colors using K-means
                n_colors = 5
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(pixels)
                
                # Get cluster centers
                colors = kmeans.cluster_centers_.astype(int)
                
                # Convert to list of lists
                return colors.tolist()
                
        except Exception as e:
            logger.error(f"Error extracting color palette: {e}")
            return []
    
    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        import hashlib
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_media_type(self, file_path: Path) -> str:
        """Determine media type from file extension"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic'}
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        
        ext = file_path.suffix.lower()
        if ext in image_extensions:
            return 'image'
        elif ext in video_extensions:
            return 'video'
        else:
            return 'unknown'
    
    def _extract_image_metadata(self, file_path: Path) -> Dict:
        """Extract image metadata using PIL and piexif"""
        from PIL import Image, ExifTags
        import piexif
        
        metadata = {}
        
        try:
            with Image.open(file_path) as img:
                # Basic info
                metadata.update({
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height
                })
                
                # EXIF data
                if hasattr(img, '_getexif') and img._getexif():
                    exif = {
                        ExifTags.TAGS.get(k, k): v
                        for k, v in img._getexif().items()
                        if k in ExifTags.TAGS
                    }
                    metadata['exif'] = exif
                    
                    # GPS data
                    if 'GPSInfo' in exif:
                        gps_data = self._extract_gps_data(exif['GPSInfo'])
                        if gps_data:
                            metadata['gps'] = gps_data
                
                # Try piexif for more comprehensive EXIF
                try:
                    exif_dict = piexif.load(str(file_path))
                    metadata['piexif'] = exif_dict
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    def _extract_gps_data(self, gps_info: Dict) -> Dict:
        """Extract GPS coordinates from EXIF GPS info"""
        try:
            def convert_to_degrees(value):
                """Convert GPS coordinates stored in EXIF to degrees"""
                d = float(value[0])
                m = float(value[1])
                s = float(value[2])
                return d + (m / 60.0) + (s / 3600.0)
            
            gps_data = {}
            
            # Latitude
            if 2 in gps_info and 1 in gps_info:
                lat = convert_to_degrees(gps_info[2])
                if gps_info[1] == b'S':
                    lat = -lat
                gps_data['latitude'] = lat
            
            # Longitude
            if 4 in gps_info and 3 in gps_info:
                lon = convert_to_degrees(gps_info[4])
                if gps_info[3] == b'W':
                    lon = -lon
                gps_data['longitude'] = lon
            
            # Altitude
            if 6 in gps_info:
                gps_data['altitude'] = float(gps_info[6])
            
            return gps_data
            
        except Exception as e:
            logger.error(f"Error extracting GPS data: {e}")
            return {}
    
    async def batch_analyze(self, file_paths: List[Path], max_concurrent: int = 4) -> List[MediaAnalysis]:
        """Analyze multiple files concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_with_semaphore(file_path):
            async with semaphore:
                return await self.analyze_media(file_path)
        
        tasks = [analyze_with_semaphore(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch analysis error: {result}")
            else:
                valid_results.append(result)
        
        return valid_results
    
    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()