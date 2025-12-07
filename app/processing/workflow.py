# app/processing/workflow.py

"""
Processing workflow and task scheduler
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path
from sqlalchemy.orm import Session
from app.db.models import ProcessingQueue, File
from app.processing.ingestion import FileIngestor
from app.processing.analysis import AnalysisPipeline
from app.processing.burst_culling import BurstCulling
from app.processing.metadata_writer import MetadataWriter
from app.processing.privacy_vault import PrivacyVault

logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Task priorities"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskType(Enum):
    """Task types"""
    INGEST = 'ingest'
    ANALYZE = 'analyze'
    BURST_CULL = 'burst_cull'
    WRITE_METADATA = 'write_metadata'
    VAULT_SCAN = 'vault_scan'
    CLEANUP = 'cleanup'


@dataclass
class WorkflowTask:
    """Workflow task definition"""
    task_type: TaskType
    priority: TaskPriority
    data: Dict[str, Any]
    created_at: datetime
    scheduled_for: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0


class TaskScheduler:
    """Schedule and manage processing tasks"""
    
    def __init__(self, config: Dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.tasks = []
        self.running = False
        self.worker_tasks = []
        
        # Initialize processors
        self.ingestor = None
        self.analysis_pipeline = None
        self.burst_culler = None
        self.metadata_writer = None
        self.privacy_vault = None
    
    async def start(self, num_workers: int = 2):
        """Start task scheduler"""
        self.running = True
        
        # Initialize processors
        with self.db_manager.get_session() as session:
            self.ingestor = FileIngestor(self.config, self.db_manager)
            self.analysis_pipeline = AnalysisPipeline(self.config, self.db_manager)
            self.burst_culler = BurstCulling(self.config, session)
            self.metadata_writer = MetadataWriter(self.config, session)
            self.privacy_vault = PrivacyVault(self.config, session)
        
        # Start analysis pipeline
        analysis_workers = self.config.get('processing', {}).get('max_workers', 4)
        await self.analysis_pipeline.start(analysis_workers)
        
        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(i))
            self.worker_tasks.append(task)
        
        logger.info(f"Started task scheduler with {num_workers} workers")
    
    async def stop(self):
        """Stop task scheduler"""
        self.running = False
        
        # Stop analysis pipeline
        if self.analysis_pipeline:
            await self.analysis_pipeline.stop()
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("Task scheduler stopped")
    
    async def _worker_loop(self, worker_id: int):
        """Worker processing loop"""
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get next task
                task = await self._get_next_task()
                if not task:
                    await asyncio.sleep(1)
                    continue
                
                # Process task
                logger.info(f"Worker {worker_id} processing task: {task.task_type}")
                success = await self._process_task(task)
                
                if not success and task.retry_count < task.max_retries:
                    # Retry task
                    task.retry_count += 1
                    task.scheduled_for = datetime.now() + timedelta(minutes=5)
                    logger.info(f"Scheduling retry {task.retry_count} for task {task.task_type}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _get_next_task(self) -> Optional[WorkflowTask]:
        """Get next task to process"""
        # Check scheduled tasks first
        now = datetime.now()
        for task in self.tasks:
            if task.scheduled_for and task.scheduled_for <= now:
                self.tasks.remove(task)
                return task
        
        # Check database queue
        with self.db_manager.get_session() as session:
            db_task = session.query(ProcessingQueue).filter(
                ProcessingQueue.status == 'pending'
            ).order_by(
                ProcessingQueue.priority.desc(),
                ProcessingQueue.created_at.asc()
            ).first()
            
            if db_task:
                # Convert to workflow task
                task_type = TaskType(db_task.task_type)
                priority = TaskPriority(db_task.priority // 10)  # Map to enum
                
                workflow_task = WorkflowTask(
                    task_type=task_type,
                    priority=priority,
                    data={'queue_id': db_task.id, 'file_id': db_task.file_id},
                    created_at=db_task.created_at or datetime.now()
                )
                
                # Update status
                db_task.status = 'processing'
                db_task.started_at = datetime.utcnow()
                session.commit()
                
                return workflow_task
        
        return None
    
    async def _process_task(self, task: WorkflowTask) -> bool:
        """Process a workflow task"""
        try:
            if task.task_type == TaskType.INGEST:
                return await self._process_ingest_task(task)
            elif task.task_type == TaskType.ANALYZE:
                return await self._process_analyze_task(task)
            elif task.task_type == TaskType.BURST_CULL:
                return await self._process_burst_cull_task(task)
            elif task.task_type == TaskType.WRITE_METADATA:
                return await self._process_write_metadata_task(task)
            elif task.task_type == TaskType.VAULT_SCAN:
                return await self._process_vault_scan_task(task)
            elif task.task_type == TaskType.CLEANUP:
                return await self._process_cleanup_task(task)
            else:
                logger.error(f"Unknown task type: {task.task_type}")
                return False
                
        except Exception as e:
            logger.error(f"Error processing task {task.task_type}: {e}")
            return False
    
    async def _process_ingest_task(self, task: WorkflowTask) -> bool:
        """Process ingestion task"""
        try:
            directory = Path(task.data.get('directory', self.config['paths']['source']))
            recursive = task.data.get('recursive', True)
            
            stats = await self.ingestor.process_directory(directory, recursive)
            logger.info(f"Ingestion completed: {stats}")
            return True
            
        except Exception as e:
            logger.error(f"Error in ingestion task: {e}")
            return False
    
    async def _process_analyze_task(self, task: WorkflowTask) -> bool:
        """Process analysis task"""
        try:
            # This is handled by the analysis pipeline
            # Just mark as completed in database
            with self.db_manager.get_session() as session:
                if 'queue_id' in task.data:
                    queue_item = session.query(ProcessingQueue).filter(
                        ProcessingQueue.id == task.data['queue_id']
                    ).first()
                    
                    if queue_item:
                        queue_item.status = 'completed'
                        queue_item.completed_at = datetime.utcnow()
                        session.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in analysis task: {e}")
            return False
    
    async def _process_burst_cull_task(self, task: WorkflowTask) -> bool:
        """Process burst culling task"""
        try:
            directory = Path(task.data.get('directory', self.config['paths']['source']))
            keep_count = task.data.get('keep_count', 1)
            action = task.data.get('action', 'mark')
            
            with self.db_manager.get_session() as session:
                self.burst_culler.db = session
                results = await self.burst_culler.process_directory(directory, keep_count, action)
            
            logger.info(f"Burst culling completed: {results}")
            return True
            
        except Exception as e:
            logger.error(f"Error in burst culling task: {e}")
            return False
    
    async def _process_write_metadata_task(self, task: WorkflowTask) -> bool:
        """Process metadata writing task"""
        try:
            if 'file_id' in task.data:
                # Single file
                success = await self.metadata_writer.write_file_metadata(task.data['file_id'])
                return success
            else:
                # Batch write
                results = await self.metadata_writer.write_all_pending()
                logger.info(f"Metadata writing completed: {results}")
                return results['failed'] == 0
                
        except Exception as e:
            logger.error(f"Error in metadata writing task: {e}")
            return False
    
    async def _process_vault_scan_task(self, task: WorkflowTask) -> bool:
        """Process vault scanning task"""
        try:
            directory = Path(task.data.get('directory', self.config['paths']['source']))
            recursive = task.data.get('recursive', True)
            
            results = await self.privacy_vault.scan_and_vault(directory, recursive)
            logger.info(f"Vault scan completed: {results}")
            return True
            
        except Exception as e:
            logger.error(f"Error in vault scan task: {e}")
            return False
    
    async def _process_cleanup_task(self, task: WorkflowTask) -> bool:
        """Process cleanup task"""
        try:
            # Clean up orphaned files
            directory = Path(self.config['paths']['source'])
            orphans = await self.ingestor.cleanup_orphans(directory)
            
            # Clean up old tasks from queue
            with self.db_manager.get_session() as session:
                cutoff = datetime.utcnow() - timedelta(days=7)
                deleted = session.query(ProcessingQueue).filter(
                    ProcessingQueue.completed_at < cutoff
                ).delete()
                session.commit()
                
                logger.info(f"Cleanup completed: {orphans} orphans, {deleted} old tasks")
                return True
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")
            return False
    
    async def schedule_task(self, task_type: TaskType, priority: TaskPriority = TaskPriority.NORMAL,
                          data: Optional[Dict] = None, delay_minutes: int = 0) -> WorkflowTask:
        """Schedule a new task"""
        scheduled_for = None
        if delay_minutes > 0:
            scheduled_for = datetime.now() + timedelta(minutes=delay_minutes)
        
        task = WorkflowTask(
            task_type=task_type,
            priority=priority,
            data=data or {},
            created_at=datetime.now(),
            scheduled_for=scheduled_for
        )
        
        self.tasks.append(task)
        logger.info(f"Scheduled task: {task_type} (priority: {priority})")
        
        return task
    
    async def schedule_ingestion(self, directory: Path, recursive: bool = True,
                               priority: TaskPriority = TaskPriority.NORMAL):
        """Schedule ingestion task"""
        return await self.schedule_task(
            task_type=TaskType.INGEST,
            priority=priority,
            data={'directory': str(directory), 'recursive': recursive}
        )
    
    async def schedule_analysis(self, file_ids: List[int], 
                              priority: TaskPriority = TaskPriority.NORMAL):
        """Schedule analysis tasks"""
        with self.db_manager.get_session() as session:
            for file_id in file_ids:
                queue_item = ProcessingQueue(
                    file_id=file_id,
                    task_type='analyze',
                    priority=priority.value * 10,  # Convert to numeric priority
                    status='pending'
                )
                session.add(queue_item)
            
            session.commit()
        
        logger.info(f"Scheduled analysis for {len(file_ids)} files")
    
    async def schedule_burst_culling(self, directory: Path, keep_count: int = 1,
                                   action: str = 'mark', 
                                   priority: TaskPriority = TaskPriority.LOW):
        """Schedule burst culling task"""
        return await self.schedule_task(
            task_type=TaskType.BURST_CULL,
            priority=priority,
            data={'directory': str(directory), 'keep_count': keep_count, 'action': action}
        )
    
    async def schedule_metadata_write(self, file_ids: Optional[List[int]] = None,
                                    priority: TaskPriority = TaskPriority.NORMAL):
        """Schedule metadata writing task"""
        data = {}
        if file_ids:
            data['file_ids'] = file_ids
        
        return await self.schedule_task(
            task_type=TaskType.WRITE_METADATA,
            priority=priority,
            data=data
        )
    
    async def schedule_vault_scan(self, directory: Path, recursive: bool = True,
                                priority: TaskPriority = TaskPriority.HIGH):
        """Schedule vault scanning task"""
        return await self.schedule_task(
            task_type=TaskType.VAULT_SCAN,
            priority=priority,
            data={'directory': str(directory), 'recursive': recursive}
        )
    
    async def schedule_cleanup(self, priority: TaskPriority = TaskPriority.LOW):
        """Schedule cleanup task"""
        return await self.schedule_task(
            task_type=TaskType.CLEANUP,
            priority=priority,
            data={}
        )
    
    async def schedule_daily_tasks(self):
        """Schedule daily maintenance tasks"""
        # Schedule cleanup at 2 AM
        cleanup_time = datetime.now().replace(hour=2, minute=0, second=0, microsecond=0)
        if datetime.now() > cleanup_time:
            cleanup_time += timedelta(days=1)
        
        delay = (cleanup_time - datetime.now()).total_seconds() / 60
        
        await self.schedule_task(
            task_type=TaskType.CLEANUP,
            priority=TaskPriority.LOW,
            delay_minutes=int(delay)
        )
        
        # Schedule metadata write for new files at 3 AM
        metadata_time = cleanup_time + timedelta(hours=1)
        delay = (metadata_time - datetime.now()).total_seconds() / 60
        
        await self.schedule_task(
            task_type=TaskType.WRITE_METADATA,
            priority=TaskPriority.NORMAL,
            delay_minutes=int(delay)
        )
        
        logger.info("Daily tasks scheduled")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        with self.db_manager.get_session() as session:
            pending = session.query(ProcessingQueue).filter(
                ProcessingQueue.status == 'pending'
            ).count()
            
            processing = session.query(ProcessingQueue).filter(
                ProcessingQueue.status == 'processing'
            ).count()
            
            completed = session.query(ProcessingQueue).filter(
                ProcessingQueue.status == 'completed'
            ).count()
        
        return {
            'running': self.running,
            'scheduled_tasks': len(self.tasks),
            'queue_pending': pending,
            'queue_processing': processing,
            'queue_completed': completed,
            'worker_count': len(self.worker_tasks),
            'analysis_status': self.analysis_pipeline.get_status() if self.analysis_pipeline else {}
        }


class ProcessingWorkflow:
    """High-level workflow management"""
    
    def __init__(self, config: Dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.scheduler = TaskScheduler(config, db_manager)
    
    async def start(self):
        """Start processing workflow"""
        await self.scheduler.start()
        
        # Schedule initial tasks
        await self.scheduler.schedule_daily_tasks()
        
        # Schedule initial ingestion
        source_dir = Path(self.config['paths']['source'])
        await self.scheduler.schedule_ingestion(source_dir, priority=TaskPriority.HIGH)
        
        logger.info("Processing workflow started")
    
    async def stop(self):
        """Stop processing workflow"""
        await self.scheduler.stop()
        logger.info("Processing workflow stopped")
    
    async def ingest_new_photos(self, directory: Optional[Path] = None):
        """Ingest new photos workflow"""
        if directory is None:
            directory = Path(self.config['paths']['source'])
        
        # 1. Schedule ingestion
        await self.scheduler.schedule_ingestion(directory, priority=TaskPriority.HIGH)
        
        # 2. Get newly ingested files
        with self.db_manager.get_session() as session:
            new_files = session.query(File).filter(
                File.processed == False,
                File.ingested_at >= datetime.utcnow() - timedelta(hours=1)
            ).all()
            
            file_ids = [f.id for f in new_files]
        
        # 3. Schedule analysis
        if file_ids:
            await self.scheduler.schedule_analysis(file_ids, priority=TaskPriority.HIGH)
        
        # 4. Schedule burst culling (lower priority)
        await self.scheduler.schedule_burst_culling(
            directory, keep_count=1, action='mark', priority=TaskPriority.LOW
        )
        
        logger.info(f"Scheduled ingestion workflow for {directory}")
    
    async def full_analysis_workflow(self, directory: Optional[Path] = None):
        """Full analysis workflow for existing photos"""
        if directory is None:
            directory = Path(self.config['paths']['source'])
        
        # 1. Find unanalyzed files
        with self.db_manager.get_session() as session:
            unanalyzed = session.query(File).filter(
                File.processed == False
            ).all()
            
            file_ids = [f.id for f in unanalyzed]
        
        # 2. Schedule analysis
        if file_ids:
            await self.scheduler.schedule_analysis(file_ids, priority=TaskPriority.NORMAL)
        
        # 3. Schedule metadata write (after analysis completes)
        await self.scheduler.schedule_metadata_write(
            file_ids, priority=TaskPriority.LOW
        )
        
        # 4. Schedule vault scan
        await self.scheduler.schedule_vault_scan(
            directory, recursive=True, priority=TaskPriority.HIGH
        )
        
        logger.info(f"Scheduled full analysis workflow for {len(file_ids)} files")
    
    async def privacy_workflow(self, directory: Optional[Path] = None):
        """Privacy protection workflow"""
        if directory is None:
            directory = Path(self.config['paths']['source'])
        
        # 1. Schedule vault scan
        await self.scheduler.schedule_vault_scan(
            directory, recursive=True, priority=TaskPriority.CRITICAL
        )
        
        # 2. Schedule metadata cleanup (remove sensitive info)
        # This would be a custom task to clean metadata
        
        logger.info(f"Scheduled privacy workflow for {directory}")
    
    async def maintenance_workflow(self):
        """Maintenance workflow"""
        # 1. Schedule cleanup
        await self.scheduler.schedule_cleanup(priority=TaskPriority.LOW)
        
        # 2. Schedule database optimization
        
        logger.info("Scheduled maintenance workflow")
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get workflow status"""
        scheduler_status = self.scheduler.get_status()
        
        with self.db_manager.get_session() as session:
            total_files = session.query(File).count()
            analyzed_files = session.query(File).filter(File.processed == True).count()
            vaulted_files = session.query(File).filter(
                File.error.like('Vaulted:%')
            ).count()
        
        return {
            'scheduler': scheduler_status,
            'files': {
                'total': total_files,
                'analyzed': analyzed_files,
                'vaulted': vaulted_files,
                'pending': total_files - analyzed_files
            },
            'workflows': {
                'ingestion_scheduled': any(
                    t.task_type == TaskType.INGEST for t in self.scheduler.tasks
                ),
                'analysis_running': scheduler_status['analysis_status'].get('running', False)
            }
        }