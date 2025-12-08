# core/intelligence_engine.py

"""
Main intelligence engine orchestrating all AI analysis
Phase-2 (GPU rewrite + HEIC + VisionImageAnalysis v2.1-gpu) – FINAL
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from app.core.local_vision import LocalVisionModels, VisionImageAnalysis
from .face_analysis import FaceAnalyzer, FaceDetection, FaceDatabase
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


# ===============================================================
# DATA STRUCTURES
# ===============================================================

@dataclass
class ImageAnalysisResult:
    caption: Optional[str]
    tags: List[str]
    objects: List[str]
    mood: Optional[str]
    contains_text: Optional[bool]
    is_sensitive: Optional[bool]


@dataclass
class MediaAnalysis:
    """
    Unified analysis object consumed by analysis_worker.
    All fields here match analysis_worker expectations exactly.
    """

    file_path: Path
    file_hash: str
    file_size: int
    media_type: str  # 'image' or 'video'

    # IMAGE ANALYSIS
    image_analysis: Optional[ImageAnalysisResult] = None

    # FACES
    face_detections: List[FaceDetection] = None
    face_clusters: List[int] = None

    # VIDEO
    video_metadata: Optional[Dict] = None
    scene_changes: List[float] = None

    # METADATA
    technical_metadata: Optional[Dict] = None
    exif_data: Optional[Dict] = None
    gps_data: Optional[Dict] = None

    # GPU AESTHETICS
    aesthetic_score: float = 0.0
    sharpness: float = 0.0
    color_palette: List[List[int]] = None

    # SYSTEM
    processing_time: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        result = asdict(self)
        result["file_path"] = str(self.file_path)

        if self.face_detections:
            result["face_detections"] = [
                {
                    "bbox": det.bbox,
                    "confidence": det.confidence,
                    "gender": det.gender,
                    "age": det.age,
                    "eye_open": det.eye_open,
                }
                for det in self.face_detections
            ]

        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


# ===============================================================
# INTELLIGENCE ENGINE (GPU)
# ===============================================================

class IntelligenceEngine:
    """
    GPU-only intelligence engine orchestrating:
        - BLIP captioning
        - CLIP embeddings + aesthetics
        - DETR object detection (lazy loaded)
        - TrOCR OCR (lazy loaded)
        - InsightFace face detection
        - HEIC-safe pipeline
    """

    def __init__(self, config: Any):
        self.config = config

        # GPU Vision Models
        self.vision = LocalVisionModels()

        # Face recognition
        self.face_analyzer = FaceAnalyzer()
        self.face_db = FaceDatabase(Path("./data/faces.db"))

        # Video
        self.video_processor = VideoProcessor()

        # Stats (optional)
        self.stats = {
            "total_processed": 0,
            "images_analyzed": 0,
            "videos_analyzed": 0,
            "faces_detected": 0,
            "errors": 0,
        }

    # ===========================================================
    # MAIN ENTRY POINT
    # ===========================================================

    async def analyze_media(self, file_path: Path) -> MediaAnalysis:
        file_path = Path(file_path)
        start_time = datetime.now()

        media_type = self._get_media_type(file_path)
        file_hash = self._compute_file_hash(file_path)
        file_size = file_path.stat().st_size

        analysis = MediaAnalysis(
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            media_type=media_type,
            face_detections=[],
            face_clusters=[],
            scene_changes=[],
            color_palette=[],
        )

        try:
            if media_type == "image":
                await self._analyze_image(file_path, analysis)
            elif media_type == "video":
                await self._analyze_video(file_path, analysis)
            else:
                analysis.error = "unsupported_media_type"
                return analysis

            # Update stats
            self.stats["total_processed"] += 1
            if media_type == "image":
                self.stats["images_analyzed"] += 1
            if media_type == "video":
                self.stats["videos_analyzed"] += 1
            self.stats["faces_detected"] += len(analysis.face_detections or [])

        except Exception as e:
            logger.error(f"[Engine] Fatal error analyzing {file_path}: {e}", exc_info=True)
            analysis.error = str(e)
            self.stats["errors"] += 1

        analysis.processing_time = (datetime.now() - start_time).total_seconds()
        return analysis

    # ===========================================================
    # IMAGE PIPELINE
    # ===========================================================

    async def _analyze_image(self, file_path: Path, analysis: MediaAnalysis):
        # -----------------------------------------------------------
        # 1) METADATA
        # -----------------------------------------------------------
        meta = self._extract_image_metadata(file_path)
        analysis.technical_metadata = meta
        analysis.exif_data = meta.get("exif", {})
        analysis.gps_data = meta.get("gps", {})

        # -----------------------------------------------------------
        # 2) GPU Vision (BLIP + CLIP + DETR + TrOCR)
        # -----------------------------------------------------------
        try:
            vision: VisionImageAnalysis = self.vision.analyze_image(file_path)

        except Exception as e:
            logger.error(
                f"[Engine] GPU vision stage failed for {file_path}: {e}",
                exc_info=True
            )
            analysis.error = "vision_stage_failure"
            return

        # SHARPNESS + PALETTE (values already normalized)
        analysis.sharpness = vision.sharpness_score
        analysis.color_palette = vision.color_palette or []

        # Aesthetic score (CLIP)
        try:
            analysis.aesthetic_score = self.vision.estimate_aesthetic_score(file_path)
        except Exception:
            analysis.aesthetic_score = 0.5

        # Create ImageAnalysisResult
        analysis.image_analysis = ImageAnalysisResult(
            caption=vision.caption,
            tags=vision.tags or [],
            objects=vision.objects or [],
            mood=vision.mood,
            contains_text=vision.contains_text,
            is_sensitive=vision.is_sensitive,
        )

        # -----------------------------------------------------------
        # 3) FACE ANALYSIS (HEIC-safe via new loader in FaceAnalyzer)
        # -----------------------------------------------------------
        try:
            faces = self.face_analyzer.detect_faces(file_path)
        except Exception as e:
            logger.error(f"[Engine] Face analysis failed: {e}")
            faces = []

        analysis.face_detections = faces

        # Clustering if multiple faces
        if len(faces) > 1:
            embeddings = [f.embedding for f in faces]
            labels = self.face_analyzer.cluster_faces(embeddings)
            analysis.face_clusters = labels.tolist()

    # ===========================================================
    # VIDEO PIPELINE
    # ===========================================================

    async def _analyze_video(self, file_path: Path, analysis: MediaAnalysis):
        # Metadata
        try:
            analysis.video_metadata = self.video_processor.get_video_metadata(file_path)
        except Exception as e:
            logger.error(f"[Video] Metadata failure: {e}")

        # Middle keyframe → temporary JPEG path
        keyframe = self.video_processor.extract_middle_keyframe(file_path)

        if keyframe:
            try:
                vision = self.vision.analyze_image(keyframe)

                analysis.image_analysis = ImageAnalysisResult(
                    caption=vision.caption,
                    tags=vision.tags or [],
                    objects=vision.objects or [],
                    mood=vision.mood,
                    contains_text=vision.contains_text,
                    is_sensitive=vision.is_sensitive,
                )

                analysis.aesthetic_score = self.vision.estimate_aesthetic_score(keyframe)
                analysis.sharpness = vision.sharpness_score
                analysis.color_palette = vision.color_palette or []

                # Face detection on keyframe
                analysis.face_detections = self.face_analyzer.detect_faces(keyframe)

                # Scene changes
                try:
                    analysis.scene_changes = self.video_processor.detect_scene_changes(file_path)
                except Exception:
                    analysis.scene_changes = []

            finally:
                try:
                    keyframe.unlink(missing_ok=True)
                except:
                    pass

        # Thumbnail
        try:
            out_thumb = Path("./data/thumbnails") / f"{file_path.stem}_thumb.jpg"
            out_thumb.parent.mkdir(parents=True, exist_ok=True)
            self.video_processor.generate_video_thumbnail(file_path, out_thumb)
        except Exception as e:
            logger.error(f"[Video] Thumbnail error: {e}")

    # ===========================================================
    # UTILITIES
    # ===========================================================

    def _compute_file_hash(self, file_path: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for blk in iter(lambda: f.read(4096), b""):
                h.update(blk)
        return h.hexdigest()

    def _get_media_type(self, file_path: Path) -> str:
        img = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".heic", ".heif"}
        vid = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".wmv"}
        ext = file_path.suffix.lower()
        if ext in img:
            return "image"
        if ext in vid:
            return "video"
        return "unknown"

    def _extract_image_metadata(self, file_path: Path) -> Dict:
        from PIL import Image, ExifTags
        import piexif

        meta = {}
        try:
            with Image.open(file_path) as img:
                meta["format"] = img.format
                meta["mode"] = img.mode
                meta["size"] = img.size

                if hasattr(img, "_getexif") and img._getexif():
                    raw = img._getexif()
                    exif = {
                        ExifTags.TAGS.get(k, k): v
                        for k, v in raw.items()
                        if k in ExifTags.TAGS
                    }
                    meta["exif"] = exif

                    if "GPSInfo" in exif:
                        gps = self._extract_gps(exif["GPSInfo"])
                        if gps:
                            meta["gps"] = gps

                try:
                    meta["piexif"] = piexif.load(str(file_path))
                except:
                    pass

        except Exception as e:
            logger.error(f"[Metadata] Failed on {file_path}: {e}")

        return meta

    def _extract_gps(self, gps_info: Dict) -> Dict:
        try:
            def deg(val):
                d, m, s = float(val[0]), float(val[1]), float(val[2])
                return d + m / 60 + s / 3600

            gps = {}
            if 2 in gps_info and 1 in gps_info:
                lat = deg(gps_info[2])
                if gps_info[1] in ("S", b"S"):
                    lat = -lat
                gps["latitude"] = lat

            if 4 in gps_info and 3 in gps_info:
                lon = deg(gps_info[4])
                if gps_info[3] in ("W", b"W"):
                    lon = -lon
                gps["longitude"] = lon

            if 6 in gps_info:
                gps["altitude"] = float(gps_info[6])

            return gps

        except Exception:
            return {}

    # ===========================================================
    # BATCH
    # ===========================================================

    async def batch_analyze(self, paths: List[Path], max_concurrent: int = 4):
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run(p):
            async with semaphore:
                return await self.analyze_media(p)

        tasks = [run(Path(p)) for p in paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"[Batch] {r}")
            else:
                out.append(r)

        return out

    def get_stats(self):
        return dict(self.stats)
