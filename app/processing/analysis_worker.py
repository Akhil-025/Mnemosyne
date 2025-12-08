# app/processing/analysis_worker.py

"""
Background worker that consumes ProcessingQueue and runs IntelligenceEngine
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy.orm import Session

from app.db.models import (
    File,
    ProcessingQueue,
    AnalysisResult,
    Embedding,
    FaceDetection as FaceDetectionModel,
)

from app.core.intelligence_engine import IntelligenceEngine

logger = logging.getLogger(__name__)

ANALYSIS_VERSION = "v2.1-gpu"

TEMP_ERROR_SUBSTRINGS = [
    "CUDA out of memory",
    "Failed to connect to",
    "HTTPConnectionPool",
    "temporarily unavailable",
    "cannot load image",            # e.g., transient file access
    "face loader: cannot read",     # in case of transient issue
]

FINAL_ERROR_SUBSTRINGS = [
    "No such file or directory",
    "unsupported image format",
    "cannot identify image file",
]

RETRY_DELAYS = [3, 9, 27]  # seconds

class AnalysisWorker:
    """
    Background analysis worker:
      - Picks pending tasks from processing_queue
      - Runs IntelligenceEngine
      - Writes analysis_result, embeddings, face_detections
    """

    def __init__(
        self,
        config: Any,
        db_manager: Any,
        intelligence: IntelligenceEngine,
        poll_interval: float = 2.0,
    ):
        self.config = config
        self.db_manager = db_manager
        self.intelligence = intelligence
        self.poll_interval = poll_interval
        self._running = False
        self.intelligence.vision.debug_aggressive_unload = True


    # ----------------------------------------------------------
    # WORKER START/STOP
    # ----------------------------------------------------------

    async def start(self):
        if self._running:
            logger.warning("[WORKER] Already running")
            return

        logger.info("[WORKER] Starting AnalysisWorker")
        self._running = True

        try:
            while self._running:
                has_task = await self._process_next_task()
                if not has_task:
                    await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("[WORKER] Cancelled")
        except Exception as e:
            logger.error(f"[WORKER] Fatal error: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info("[WORKER] Stopped AnalysisWorker")

    async def stop(self):
        logger.info("[WORKER] Stop requested")
        self._running = False

    # ----------------------------------------------------------
    # POLL QUEUE
    # ----------------------------------------------------------

    async def _process_next_task(self) -> bool:
        """
        Fetch and process a single pending task.
        Returns True if processed, False if queue empty.
        """
        with self.db_manager.get_session() as session:
            row = (
                session.query(
                    ProcessingQueue.id,
                    ProcessingQueue.file_id,
                    ProcessingQueue.task_type,
                )
                .filter(ProcessingQueue.status == "pending")
                .order_by(
                    ProcessingQueue.priority.desc(),
                    ProcessingQueue.created_at.asc(),
                )
                .first()
            )

            if not row:
                return False

            queue_id = row.id
            file_id = row.file_id
            task_type = row.task_type

            # Get file path without loading full ORM object
            fp = session.query(File.file_path).filter(File.id == file_id).first()
            if not fp:
                self._fail_queue_item(session, queue_id, "Missing file entry")
                return True

            file_path = Path(fp.file_path)

            if not file_path.exists():
                self._fail_queue_item(session, queue_id, "File missing on disk")
                file_obj = session.get(File, file_id)
                if file_obj:
                    file_obj.error = "File missing on disk"
                session.commit()
                return True

            # Mark item as processing
            item = session.get(ProcessingQueue, queue_id)
            item.status = "processing"
            item.started_at = datetime.utcnow()
            session.commit()

        # ------------------------------------
        # Dispatch work outside session
        # ------------------------------------
        if task_type == "analyze":
            await self._handle_analyze_task(queue_id, file_id, file_path)
        else:
            logger.warning(f"[WORKER] Unknown task type '{task_type}'")
            with self.db_manager.get_session() as session:
                self._fail_queue_item(session, queue_id, f"Unknown task type {task_type}")

        return True

    # ----------------------------------------------------------
    # ANALYSIS TASK
    # ----------------------------------------------------------

    async def _handle_analyze_task(self, queue_id: int, file_id: int, file_path: Path):
        logger.info(f"[WORKER] ANALYZE file_id={file_id} path={file_path}")

        try:
            # The IntelligenceEngine should internally call LocalVisionModels.analyze_image,
            # which now includes sharpness_score + color_palette.
            analysis = await self.intelligence.analyze_media(file_path)
        except Exception as e:
            logger.error(f"[WORKER] Engine crashed on file {file_id}: {e}")
            with self.db_manager.get_session() as session:
                self._fail_queue_item(session, queue_id, str(e))
            return

        # Save results
        with self.db_manager.get_session() as session:
            file_obj = session.get(File, file_id)
            item = session.get(ProcessingQueue, queue_id)

            if not file_obj or not item:
                logger.error("[WORKER] File or queue item vanished mid-analysis")
                return

            # 1) Write main analysis_result
            self._upsert_analysis_result(session, file_id, analysis)

            # 2) Write caption/tags CLIP embedding
            await self._create_clip_caption_embedding(session, file_id, analysis)

            # 3) Write face embeddings
            self._store_face_detections(session, file_id, analysis)

            # File update
            file_obj.processed = True
            file_obj.analyzed_at = datetime.utcnow()
            file_obj.error = analysis.error

            # Queue update
            item.status = "completed" if not analysis.error else "failed"
            item.completed_at = datetime.utcnow()
            if analysis.error:
                item.error = analysis.error

            session.commit()

        # Let vision stack clean up DETR/TrOCR if idle
        if hasattr(self.intelligence, "vision"):
            self.intelligence.vision.maybe_unload_idle_models()

        logger.info(f"[WORKER] DONE ANALYZE file_id={file_id}")

    # ----------------------------------------------------------
    # FAIL QUEUE
    # ----------------------------------------------------------

    def _fail_queue_item(self, session: Session, queue_id: int, message: str):
        item = session.get(ProcessingQueue, queue_id)
        if item:
            item.status = "failed"
            item.completed_at = datetime.utcnow()
            item.error = message
        session.commit()
        logger.error(f"[WORKER] FAILED queue_id={queue_id}: {message}")

    # ----------------------------------------------------------
    # WRITE ANALYSIS RESULT
    # ----------------------------------------------------------

    def _upsert_analysis_result(self, session: Session, file_id: int, analysis: Any):
        existing = (
            session.query(AnalysisResult)
            .filter(AnalysisResult.file_id == file_id)
            .first()
        )

        if existing is None:
            existing = AnalysisResult(file_id=file_id)
            session.add(existing)

        img = analysis.image_analysis

        if img:
            existing.caption = img.caption
            existing.tags = img.tags
            existing.objects = img.objects
            existing.mood = img.mood
            existing.contains_text = img.contains_text
            existing.is_sensitive = img.is_sensitive

        existing.aesthetic_score = analysis.aesthetic_score
        existing.sharpness = analysis.sharpness
        existing.color_palette = analysis.color_palette

        if analysis.video_metadata:
            existing.video_duration = analysis.video_metadata.get("duration")

        existing.scene_changes = analysis.scene_changes or []

        if analysis.gps_data:
            existing.latitude = analysis.gps_data.get("latitude")
            existing.longitude = analysis.gps_data.get("longitude")
            existing.altitude = analysis.gps_data.get("altitude")

        existing.analysis_version = ANALYSIS_VERSION
        existing.analyzed_at = datetime.utcnow()

    # ----------------------------------------------------------
    # CLIP TEXT EMBEDDINGS  (replacing Ollama embeddings)
    # ----------------------------------------------------------

    async def _create_clip_caption_embedding(self, session: Session, file_id: int, analysis: Any):
        img = analysis.image_analysis
        if not img:
            return

        # Build text representation
        text = " ".join(
            [t for t in ([img.caption] + (img.tags or [])) if t]
        ).strip()

        if not text:
            return

        try:
            vec = self.intelligence.vision.get_clip_text_embedding(text)
        except Exception as e:
            logger.error(f"[WORKER] CLIP text embedding failed for file_id={file_id}: {e}")
            return

        if vec is None or vec.shape[0] == 0:
            return

        emb = Embedding(
            file_id=file_id,
            embedding_type="caption_clip",
            embedding_vector=vec.astype(np.float32).tobytes(),
            dimensions=int(vec.shape[0]),
            model_name="clip-vit-large-patch14-text",
            generated_at=datetime.utcnow(),
        )
        session.add(emb)

    # ----------------------------------------------------------
    # FACE DETECTIONS → DB
    # ----------------------------------------------------------

    def _store_face_detections(self, session: Session, file_id: int, analysis: Any):
        if not analysis.face_detections:
            return

        for det in analysis.face_detections:
            emb = np.asarray(det.embedding, dtype=np.float32)
            x1, y1, x2, y2 = det.bbox

            row = FaceDetectionModel(
                file_id=file_id,
                person_id=None,
                bbox_x1=int(x1),
                bbox_y1=int(y1),
                bbox_x2=int(x2),
                bbox_y2=int(y2),
                embedding=emb.tobytes(),
                embedding_model="insightface",
                confidence=float(det.confidence),
                gender=det.gender,
                age=int(det.age) if det.age else None,
                eye_open_prob=float(det.eye_open) if det.eye_open else None,
                cluster_id=None,
            )
            session.add(row)

    def classify_exception(e: Exception) -> str:
        """
        Returns: 'temp' | 'final'
        """
        msg = str(e)
        for s in FINAL_ERROR_SUBSTRINGS:
            if s in msg:
                return "final"
        for s in TEMP_ERROR_SUBSTRINGS:
            if s in msg:
                return "temp"
        # default: treat as temp so we retry a bit
        return "temp"
    
    def run_once(self):
        session = self.session_factory()
        try:
            # pull next queued item
            item = (
                session.query(ProcessingQueue)
                .filter_by(status="queued")
                .order_by(ProcessingQueue.created_at)
                .first()
            )
            if not item:
                return False

            item.status = "processing"
            session.commit()

            try:
                self._process_file(session, item.file_id)
                item.status = "success"
                item.last_error = None
                session.commit()
                return True
            except Exception as e:
                kind = self.classify_exception(e)
                self.logger.exception("Error processing file_id=%s: %s", item.file_id, e)

                item.retry_count = (item.retry_count or 0) + 1
                item.last_error = str(e)

                if kind == "final":
                    item.status = "failed_final"
                else:
                    if item.retry_count >= self.max_retries:
                        item.status = "failed_final"
                    else:
                        item.status = "failed_temp"

                session.commit()
                return True
        finally:
            session.close()

    def next_allowed_time(retry_count: int) -> datetime:
        idx = min(retry_count - 1, len(RETRY_DELAYS) - 1)
        return datetime.utcnow() + timedelta(seconds=RETRY_DELAYS[idx])