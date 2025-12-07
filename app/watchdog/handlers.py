# app/watchdog/handlers.py

"""
Event handlers for file system monitoring
"""
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime

from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
    DirCreatedEvent,
    DirModifiedEvent,
    DirDeletedEvent,
    DirMovedEvent
)

from .events import WatchdogEvent, EventType
from .patterns import PatternFilter
from .debounce import EventDebouncer

logger = logging.getLogger(__name__)


class EventHandler(FileSystemEventHandler):
    """
    Base event handler for file system events
    """
    
    def __init__(self, processor: Any = None, 
                 pattern_filter: Optional[PatternFilter] = None):
        """
        Initialize event handler
        
        Args:
            processor: File processor instance
            pattern_filter: Pattern filter for ignoring files
        """
        self.processor = processor
        self.pattern_filter = pattern_filter or PatternFilter()
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'events_ignored': 0,
            'last_event': None,
        }
        
        # Callbacks
        self.callbacks = {
            'on_event': [],
            'on_error': [],
        }
    
    def on_any_event(self, event):
        """Handle any file system event"""
        self.stats['events_received'] += 1
        self.stats['last_event'] = datetime.now()
        
        try:
            # Convert to our event type
            watchdog_event = self._convert_event(event)
            if watchdog_event:
                # Check if should be ignored
                if self.pattern_filter.should_ignore(watchdog_event.src_path):
                    self.stats['events_ignored'] += 1
                    logger.debug(f"Ignoring event for {watchdog_event.src_path}")
                    return
                
                # Process event
                self._handle_event(watchdog_event)
                self.stats['events_processed'] += 1
                
                # Call callbacks
                self._call_callbacks('on_event', watchdog_event)
                
        except Exception as e:
            logger.error(f"Error handling event: {e}")
            self._call_callbacks('on_error', e)
    
    def _convert_event(self, event) -> Optional[WatchdogEvent]:
        """Convert watchdog event to our internal format"""
        if isinstance(event, (FileCreatedEvent, DirCreatedEvent)):
            event_type = EventType.CREATED
        elif isinstance(event, (FileModifiedEvent, DirModifiedEvent)):
            event_type = EventType.MODIFIED
        elif isinstance(event, (FileDeletedEvent, DirDeletedEvent)):
            event_type = EventType.DELETED
        elif isinstance(event, (FileMovedEvent, DirMovedEvent)):
            event_type = EventType.MOVED
        else:
            # Unknown event type
            return None
        
        # Get paths
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path) if hasattr(event, 'dest_path') else None
        
        # Determine if it's a directory
        is_directory = isinstance(event, (DirCreatedEvent, DirModifiedEvent, 
                                         DirDeletedEvent, DirMovedEvent))
        
        return WatchdogEvent(
            event_type=event_type,
            src_path=src_path,
            dest_path=dest_path,
            is_directory=is_directory
        )
    
    def _handle_event(self, event: WatchdogEvent):
        """Handle converted event (to be overridden by subclasses)"""
        logger.debug(f"Base handler received: {event}")
    
    def register_callback(self, callback_type: str, callback: Callable):
        """Register a callback"""
        if callback_type in self.callbacks:
            self.callbacks[callback_type].append(callback)
        else:
            logger.warning(f"Unknown callback type: {callback_type}")
    
    def _call_callbacks(self, callback_type: str, *args, **kwargs):
        """Call registered callbacks"""
        if callback_type not in self.callbacks:
            return
        
        for callback in self.callbacks[callback_type]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback {callback_type}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics"""
        return self.stats.copy()


class FileEventHandler(EventHandler):
    """
    Enhanced event handler with debouncing and file processing
    """
    
    def __init__(self, processor: Any = None,
                 pattern_filter: Optional[PatternFilter] = None,
                 debouncer: Optional[EventDebouncer] = None):
        """
        Initialize file event handler
        
        Args:
            processor: File processor instance
            pattern_filter: Pattern filter
            debouncer: Event debouncer
        """
        super().__init__(processor, pattern_filter)
        self.debouncer = debouncer
        
        # File type specific handling
        self.file_extensions = {
            'image': {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
                     '.webp', '.heic', '.heif', '.raw', '.nef', '.cr2', '.arw'},
            'video': {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv',
                     '.m4v', '.mpg', '.mpeg', '.3gp', '.mts', '.m2ts'},
            'audio': {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'},
        }
        
        # Event type priorities
        self.event_priorities = {
            EventType.CREATED: 10,
            EventType.MODIFIED: 5,
            EventType.MOVED: 8,
            EventType.DELETED: 3,
        }
        
        logger.info("FileEventHandler initialized")
    
    def _handle_event(self, event: WatchdogEvent):
        """Handle converted event with debouncing"""
        # Skip directories for now (focus on files)
        if event.is_directory:
            logger.debug(f"Skipping directory event: {event.src_path}")
            return
        
        # Check if file type is supported
        if not self._is_supported_file(event.src_path):
            logger.debug(f"Skipping unsupported file type: {event.src_path}")
            return
        
        # Add to debouncer if available
        if self.debouncer:
            # Add priority based on event type
            event_priority = self.event_priorities.get(event.event_type, 5)
            
            # Add to debouncer (async but we're in sync context)
            # We'll use asyncio to schedule this
            import asyncio
            asyncio.create_task(self._add_to_debouncer(event))
        else:
            # Process immediately
            self._process_event_immediate(event)
    
    def _run_async(self, coro):
        """Safely schedule async coroutine from watchdog thread"""
        try:
            loop = asyncio.get_running_loop()
            asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError:
            # No running loop (shutdown) — ignore silently
            pass

    def _add_to_debouncer(self, event: WatchdogEvent):
        """Thread-safe wrapper (watchdog thread → asyncio loop)"""
        if self.debouncer:
            self._run_async(self.debouncer.add_event(event))
    
    def _process_event_immediate(self, event: WatchdogEvent):
        """Process event immediately (without debouncing)"""
        try:
            if event.event_type == EventType.CREATED:
                self._handle_file_created(event)
            elif event.event_type == EventType.MODIFIED:
                self._handle_file_modified(event)
            elif event.event_type == EventType.MOVED:
                self._handle_file_moved(event)
            elif event.event_type == EventType.DELETED:
                self._handle_file_deleted(event)
        except Exception as e:
            logger.error(f"Error processing event {event}: {e}")
    
    def _handle_file_created(self, event: WatchdogEvent):
        """Handle file creation event"""
        logger.info(f"File created: {event.src_path}")
        
        # Process file if we have a processor
        if self.processor:
            try:
                # Use async processor if available
                if hasattr(self.processor, 'process_single_file'):
                    import asyncio
                    asyncio.create_task(
                        self.processor.process_single_file(event.src_path)
                    )
                else:
                    # Sync processing
                    self.processor.process_single_file(event.src_path)
            except Exception as e:
                logger.error(f"Error processing created file {event.src_path}: {e}")
    
    def _handle_file_modified(self, event: WatchdogEvent):
        """Handle file modification event"""
        logger.debug(f"File modified: {event.src_path}")
        
        # Only re-process certain file types on modification
        if self._should_reprocess_on_modify(event.src_path):
            logger.info(f"Re-processing modified file: {event.src_path}")
            
            if self.processor:
                try:
                    if hasattr(self.processor, 'process_single_file'):
                        import asyncio
                        asyncio.create_task(
                            self.processor.process_single_file(event.src_path)
                        )
                except Exception as e:
                    logger.error(f"Error processing modified file {event.src_path}: {e}")
    
    def _handle_file_moved(self, event: WatchdogEvent):
        """Handle file move/rename event"""
        logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
        
        # Process destination if it's a supported file
        if event.dest_path and self._is_supported_file(event.dest_path):
            if self.processor:
                try:
                    if hasattr(self.processor, 'process_single_file'):
                        import asyncio
                        asyncio.create_task(
                            self.processor.process_single_file(event.dest_path)
                        )
                except Exception as e:
                    logger.error(f"Error processing moved file {event.dest_path}: {e}")
    
    def _handle_file_deleted(self, event: WatchdogEvent):
        """Handle file deletion event"""
        logger.info(f"File deleted: {event.src_path}")
        
        # Update database or perform cleanup
        # This would typically be handled by the processor
    
    def _is_supported_file(self, path: Path) -> bool:
        """Check if file type is supported"""
        if not path.is_file():
            return False
        
        # Check file extension
        file_ext = path.suffix.lower()
        
        # Check all supported extensions
        for extensions in self.file_extensions.values():
            if file_ext in extensions:
                return True
        
        return False
    
    def _should_reprocess_on_modify(self, path: Path) -> bool:
        """Determine if file should be re-processed on modification"""
        # Only re-process RAW files and documents on modification
        file_ext = path.suffix.lower()
        
        # RAW files might be updated by editing software
        raw_extensions = {'.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2', '.dng'}
        
        # Documents that might get metadata updates
        document_extensions = {'.pdf', '.doc', '.docx', '.xls', '.xlsx'}
        
        return file_ext in raw_extensions or file_ext in document_extensions
    
    def get_file_type(self, path: Path) -> Optional[str]:
        """Get file type category"""
        file_ext = path.suffix.lower()
        
        for file_type, extensions in self.file_extensions.items():
            if file_ext in extensions:
                return file_type
        
        return None


class BatchEventHandler(EventHandler):
    """
    Event handler that batches events for bulk processing
    """
    
    def __init__(self, processor: Any = None,
                 pattern_filter: Optional[PatternFilter] = None,
                 batch_size: int = 100,
                 batch_timeout: float = 5.0):
        """
        Initialize batch event handler
        
        Args:
            processor: File processor instance
            pattern_filter: Pattern filter
            batch_size: Maximum batch size
            batch_timeout: Timeout for batch collection
        """
        super().__init__(processor, pattern_filter)
        
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batch collection
        self.current_batch: List[WatchdogEvent] = []
        self.batch_timer = None
        self.last_batch_time = None
        
        # Statistics
        self.batch_stats = {
            'batches_processed': 0,
            'total_batched_events': 0,
            'average_batch_size': 0,
        }
        
        logger.info(f"BatchEventHandler initialized (batch_size={batch_size})")
    
    def _handle_event(self, event: WatchdogEvent):
        """Add event to current batch"""
        # Skip directories
        if event.is_directory:
            return
        
        # Add to batch
        self.current_batch.append(event)
        
        # Check if batch is full
        if len(self.current_batch) >= self.batch_size:
            self._process_batch()
        
        # Start/restart batch timer
        self._start_batch_timer()
    
    def _start_batch_timer(self):
        """Start or restart batch timeout timer"""
        if self.batch_timer:
            self.batch_timer.cancel()
        
        import asyncio
        self.batch_timer = asyncio.get_event_loop().call_later(
            self.batch_timeout,
            self._process_batch_timer
        )
    
    def _process_batch_timer(self):
        """Process batch when timer expires"""
        if self.current_batch:
            self._process_batch()
    
    def _process_batch(self):
        """Process current batch of events"""
        if not self.current_batch:
            return
        
        # Take copy of current batch
        batch_to_process = self.current_batch.copy()
        self.current_batch.clear()
        
        # Cancel timer
        if self.batch_timer:
            self.batch_timer.cancel()
            self.batch_timer = None
        
        # Process batch
        logger.info(f"Processing batch of {len(batch_to_process)} events")
        
        # Group events by type and path
        grouped_events = self._group_events(batch_to_process)
        
        # Process each group
        for event_group in grouped_events:
            self._process_event_group(event_group)
        
        # Update statistics
        self.batch_stats['batches_processed'] += 1
        self.batch_stats['total_batched_events'] += len(batch_to_process)
        
        # Update average batch size
        total_batches = self.batch_stats['batches_processed']
        total_events = self.batch_stats['total_batched_events']
        if total_batches > 0:
            self.batch_stats['average_batch_size'] = total_events / total_batches
        
        self.last_batch_time = datetime.now()
        
        logger.debug(f"Batch processed: {len(batch_to_process)} events")
    
    def _group_events(self, events: List[WatchdogEvent]) -> List[List[WatchdogEvent]]:
        """Group events by file path and type"""
        groups = {}
        
        for event in events:
            # Create group key
            key = (event.src_path, event.event_type)
            
            if key not in groups:
                groups[key] = []
            
            groups[key].append(event)
        
        # Convert to list of groups
        return list(groups.values())
    
    def _process_event_group(self, events: List[WatchdogEvent]):
        """Process a group of related events"""
        if not events:
            return
        
        # Get the most recent event in group
        latest_event = max(events, key=lambda e: e.timestamp)
        
        # Process based on event type
        if latest_event.event_type == EventType.CREATED:
            self._handle_batch_file_created(latest_event, events)
        elif latest_event.event_type == EventType.MODIFIED:
            self._handle_batch_file_modified(latest_event, events)
        elif latest_event.event_type == EventType.MOVED:
            self._handle_batch_file_moved(latest_event, events)
        elif latest_event.event_type == EventType.DELETED:
            self._handle_batch_file_deleted(latest_event, events)
    
    def _handle_batch_file_created(self, event: WatchdogEvent, all_events: List[WatchdogEvent]):
        """Handle batch of file creation events"""
        logger.info(f"Batch file created: {event.src_path} ({len(all_events)} events)")
        
        if self.processor and hasattr(self.processor, 'process_single_file'):
            import asyncio
            asyncio.create_task(
                self.processor.process_single_file(event.src_path)
            )
    
    def _handle_batch_file_modified(self, event: WatchdogEvent, all_events: List[WatchdogEvent]):
        """Handle batch of file modification events"""
        if len(all_events) > 1:
            logger.debug(f"File modified multiple times: {event.src_path} ({len(all_events)} events)")
        
        # Only process if significant number of modifications
        if len(all_events) >= 3:  # Arbitrary threshold
            logger.info(f"Re-processing frequently modified file: {event.src_path}")
            
            if self.processor and hasattr(self.processor, 'process_single_file'):
                import asyncio
                asyncio.create_task(
                    self.processor.process_single_file(event.src_path)
                )
    
    def _handle_batch_file_moved(self, event: WatchdogEvent, all_events: List[WatchdogEvent]):
        """Handle batch of file move events"""
        logger.info(f"Batch file moved: {event.src_path} -> {event.dest_path}")
        
        if event.dest_path and self.processor and hasattr(self.processor, 'process_single_file'):
            import asyncio
            asyncio.create_task(
                self.processor.process_single_file(event.dest_path)
            )
    
    def _handle_batch_file_deleted(self, event: WatchdogEvent, all_events: List[WatchdogEvent]):
        """Handle batch of file deletion events"""
        logger.info(f"Batch file deleted: {event.src_path} ({len(all_events)} events)")
        # Deletion handling would go here
    
    def flush_batch(self):
        """Force processing of current batch"""
        if self.current_batch:
            self._process_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics including batch stats"""
        stats = super().get_stats()
        stats.update({
            'batch': self.batch_stats.copy(),
            'current_batch_size': len(self.current_batch),
            'last_batch_time': self.last_batch_time,
        })
        return stats