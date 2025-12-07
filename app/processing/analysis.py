# app/processing/analysis.py

"""
AI analysis pipeline for media files
"""
import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from sqlalchemy.orm import Session

from app.db.models import File, AnalysisResult, ProcessingQueue, FaceDetection, Embedding
from app.core.intelligence_engine import IntelligenceEngine, MediaAnalysis
from app.core.ollama_client import OllamaClient
from app.core.face_analysis import FaceAnalyzer, FaceDatabase

logger = logging.getLogger(__name__)


class AnalysisWorker:
    """Worker for processing analysis tasks"""
    
    def __init__(self, config: Dict, db_session: Session):
        self.config = config
        self.db = db_session
        self.intelligence = IntelligenceEngine(config)
        self.face_analyzer = FaceAnalyzer()
        self.face_db = FaceDatabase(Path('./data/faces.db'))
        self.ollama = OllamaClient()
        
        # Statistics
        self.stats = {
            'processed': 0,
            'errors': 0,
            'total_time': 0.0
        }
    
    async def process_task(self, queue_item: ProcessingQueue) -> bool:
        """Process a single analysis task"""
        start_time = datetime.now()
        
        try:
            # Get file record
            file_record = self.db.query(File).filter(File.id == queue_item.file_id).first()
            if not file_record:
                logger.error(f"File not found: ID {queue_item.file_id}")
                queue_item.status = 'failed'
                queue_item.error = "File not found"
                return False
            
            file_path = Path(file_record.file_path)
            if not file_path.exists():
                logger.error(f"File not on disk: {file_path}")
                queue_item.status = 'failed'
                queue_item.error = "File not found on disk"
                return False
            
            # Update queue status
            queue_item.status = 'processing'
            queue_item.started_at = datetime.utcnow()
            self.db.commit()
            
            # Run analysis
            analysis = await self.intelligence.analyze_media(file_path)
            
            # Store results
            await self._store_analysis_results(file_record.id, analysis)
            
            # Generate and store embeddings
            await self._generate_embeddings(file_record.id, analysis)
            
            # Process faces if detected
            if analysis.face_detections:
                await self._process_faces(file_record.id, file_path, analysis.face_detections)
            
            # Update file record
            file_record.analyzed_at = datetime.utcnow()
            file_record.processed = True
            
            # Update queue item
            queue_item.status = 'completed'
            queue_item.completed_at = datetime.utcnow()
            
            # Update statistics
            self.stats['processed'] += 1
            self.stats['total_time'] += (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Analyzed: {file_path} (ID: {file_record.id})")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing file ID {queue_item.file_id}: {e}")
            queue_item.status = 'failed'
            queue_item.error = str(e)
            self.stats['errors'] += 1
            return False
    
    async def _store_analysis_results(self, file_id: int, analysis: MediaAnalysis):
        """Store analysis results in database"""
        # Create or update analysis result
        existing = self.db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id
        ).first()
        
        if existing:
            result = existing
        else:
            result = AnalysisResult(file_id=file_id)
        
        # Update with analysis data
        if analysis.image_analysis:
            result.caption = analysis.image_analysis.caption
            result.tags = json.dumps(analysis.image_analysis.tags)
            result.objects = json.dumps(analysis.image_analysis.objects)
            result.mood = analysis.image_analysis.mood
            result.contains_text = analysis.image_analysis.contains_text
            result.is_sensitive = analysis.image_analysis.is_sensitive
        
        # Technical scores
        result.aesthetic_score = analysis.aesthetic_score
        result.sharpness = analysis.sharpness
        
        if analysis.color_palette:
            result.color_palette = json.dumps(analysis.color_palette)
        
        # Video specific
        if analysis.video_metadata:
            result.video_duration = analysis.video_metadata.get('duration')
        
        if analysis.scene_changes:
            result.scene_changes = json.dumps(analysis.scene_changes)
        
        # GPS data
        if analysis.gps_data:
            result.latitude = analysis.gps_data.get('latitude')
            result.longitude = analysis.gps_data.get('longitude')
            result.altitude = analysis.gps_data.get('altitude')
        
        result.analysis_version = "1.0.0"
        result.analyzed_at = datetime.utcnow()
        
        if not existing:
            self.db.add(result)
        
        self.db.commit()
    
    async def _generate_embeddings(self, file_id: int, analysis: MediaAnalysis):
        """Generate and store embeddings for search"""
        # Generate embedding from caption
        if analysis.image_analysis and analysis.image_analysis.caption:
            caption = analysis.image_analysis.caption
            embedding = await self.ollama.generate_embedding(caption)
            
            if embedding:
                await self._store_embedding(file_id, 'caption', embedding, 'nomic-embed-text')
        
        # Generate embedding from tags
        if analysis.image_analysis and analysis.image_analysis.tags:
            tags_text = ', '.join(analysis.image_analysis.tags)
            embedding = await self.ollama.generate_embedding(tags_text)
            
            if embedding:
                await self._store_embedding(file_id, 'tags', embedding, 'nomic-embed-text')
    
    async def _store_embedding(self, file_id: int, embedding_type: str, 
                              embedding: List[float], model_name: str):
        """Store embedding in database"""
        # Convert to numpy array and serialize
        embedding_array = np.array(embedding, dtype=np.float32)
        embedding_blob = embedding_array.tobytes()
        
        # Check if embedding already exists
        existing = self.db.query(Embedding).filter(
            Embedding.file_id == file_id,
            Embedding.embedding_type == embedding_type
        ).first()
        
        if existing:
            existing.embedding_blob = embedding_blob
            existing.model_name = model_name
            existing.dimensions = len(embedding)
        else:
            new_embedding = Embedding(
                file_id=file_id,
                embedding_type=embedding_type,
                embedding_blob=embedding_blob,
                dimensions=len(embedding),
                model_name=model_name
            )
            self.db.add(new_embedding)
        
        self.db.commit()
    
    async def _process_faces(self, file_id: int, file_path: Path, 
                           face_detections: List[Any]):
        """Process detected faces"""
        for i, detection in enumerate(face_detections):
            # Create face detection record
            face_record = FaceDetection(
                file_id=file_id,
                bbox_x1=detection.bbox[0],
                bbox_y1=detection.bbox[1],
                bbox_x2=detection.bbox[2],
                bbox_y2=detection.bbox[3],
                embedding=detection.embedding.tobytes() if hasattr(detection.embedding, 'tobytes') else detection.embedding,
                embedding_model='insightface',
                confidence=detection.confidence,
                gender=detection.gender,
                age=detection.age,
                eye_open_prob=detection.eye_open
            )
            
            # Try to match with existing person
            if hasattr(detection, 'embedding'):
                similar_persons = self.face_db.search_person(detection.embedding, threshold=0.6)
                if similar_persons:
                    face_record.person_id = similar_persons[0][0]
                else:
                    # Create new cluster
                    cluster_id = self.face_db.add_unknown_cluster([detection.embedding])
                    face_record.cluster_id = cluster_id
            
            self.db.add(face_record)
        
        self.db.commit()


class AnalysisPipeline:
    """Orchestrate analysis pipeline with multiple workers"""
    
    def __init__(self, config: Dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.workers = []
        self.running = False
        self.stats = {
            'total_tasks': 0,
            'completed': 0,
            'failed': 0,
            'active_workers': 0
        }
    
    async def start(self, num_workers: int = 4):
        """Start analysis pipeline with specified number of workers"""
        self.running = True
        self.stats['active_workers'] = num_workers
        
        # Create workers
        for i in range(num_workers):
            worker = AnalysisWorker(self.config, self.db_manager.get_session())
            self.workers.append(worker)
        
        # Start worker tasks
        worker_tasks = []
        for worker in self.workers:
            task = asyncio.create_task(self._worker_loop(worker))
            worker_tasks.append(task)
        
        logger.info(f"Started analysis pipeline with {num_workers} workers")
        
        # Wait for all workers to complete
        await asyncio.gather(*worker_tasks)
    
    async def stop(self):
        """Stop analysis pipeline"""
        self.running = False
        logger.info("Stopping analysis pipeline...")
    
    async def _worker_loop(self, worker: AnalysisWorker):
        """Worker processing loop"""
        while self.running:
            try:
                # Get next task from queue
                task = await self._get_next_task()
                if not task:
                    # No tasks available, wait a bit
                    await asyncio.sleep(1)
                    continue
                
                # Process task
                self.stats['total_tasks'] += 1
                success = await worker.process_task(task)
                
                if success:
                    self.stats['completed'] += 1
                else:
                    self.stats['failed'] += 1
                
                # Update statistics
                await self._update_stats(worker)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(1)
    
    async def _get_next_task(self) -> Optional[ProcessingQueue]:
        """Get next task from processing queue"""
        with self.db_manager.get_session() as session:
            # Get highest priority pending task
            task = session.query(ProcessingQueue).filter(
                ProcessingQueue.status == 'pending'
            ).order_by(
                ProcessingQueue.priority.desc(),
                ProcessingQueue.created_at.asc()
            ).first()
            
            if task:
                return task
            
            return None
    
    async def _update_stats(self, worker: AnalysisWorker):
        """Update pipeline statistics"""
        # Combine worker stats
        total_processed = sum(w.stats['processed'] for w in self.workers)
        total_errors = sum(w.stats['errors'] for w in self.workers)
        total_time = sum(w.stats['total_time'] for w in self.workers)
        
        self.stats.update({
            'total_processed': total_processed,
            'total_errors': total_errors,
            'avg_time_per_task': total_time / total_processed if total_processed > 0 else 0
        })
    
    async def process_single_file(self, file_path: Path) -> Optional[MediaAnalysis]:
        """Process a single file directly (bypassing queue)"""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        with self.db_manager.get_session() as session:
            worker = AnalysisWorker(self.config, session)
            
            # Create temporary queue item
            file_record = session.query(File).filter(
                File.file_path == str(file_path)
            ).first()
            
            if not file_record:
                logger.error(f"File not in database: {file_path}")
                return None
            
            queue_item = ProcessingQueue(
                file_id=file_record.id,
                task_type='analyze',
                priority=10,  # High priority
                status='processing'
            )
            
            session.add(queue_item)
            session.commit()
            
            # Process
            success = await worker.process_task(queue_item)
            
            if success:
                # Get analysis results
                analysis = await self.intelligence.analyze_media(file_path)
                return analysis
            
            return None
    
    def get_status(self) -> Dict:
        """Get pipeline status"""
        with self.db_manager.get_session() as session:
            # Count pending tasks
            pending = session.query(ProcessingQueue).filter(
                ProcessingQueue.status == 'pending'
            ).count()
            
            processing = session.query(ProcessingQueue).filter(
                ProcessingQueue.status == 'processing'
            ).count()
            
            return {
                'running': self.running,
                'active_workers': self.stats['active_workers'],
                'pending_tasks': pending,
                'processing_tasks': processing,
                **self.stats
            }