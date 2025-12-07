# app/watchdog/monitor.py

"""
Main file system monitor for Mnemosyne
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEvent, FileSystemEventHandler

from .debounce import EventDebouncer
from .patterns import PatternFilter
from .handlers import FileEventHandler

logger = logging.getLogger(__name__)


class EventType(Enum):
    """File system event types"""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    CLOSED = "closed"  # For write completion detection


@dataclass
class WatchdogEvent:
    """Enhanced file system event"""
    event_type: EventType
    src_path: Path
    dest_path: Optional[Path] = None
    is_directory: bool = False
    timestamp: datetime = None
    file_size: Optional[int] = None
    file_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def __str__(self):
        if self.dest_path:
            return f"{self.event_type.value}: {self.src_path} -> {self.dest_path}"
        return f"{self.event_type.value}: {self.src_path}"


class FileMonitor:
    """
    Main file system monitor with debouncing and filtering
    """
    
    def __init__(self, config: Dict, processor: Any = None):
        """
        Initialize file monitor
        
        Args:
            config: Configuration dictionary
            processor: File processor instance (e.g., FileIngestor)
        """
        self.config = config
        self.processor = processor
        
        # Watchdog configuration
        watchdog_config = config.watchdog

        self.recursive = watchdog_config.recursive
        self.debounce_time = watchdog_config.debounce_time
        self.batch_size = getattr(watchdog_config, "batch_size", 100)
        self.max_events = getattr(watchdog_config, "max_events", 10000)
        
        # Components
        self.observer = None
        self.debouncer = EventDebouncer(
            debounce_time=self.debounce_time,
            max_events=self.max_events,
            batch_size=self.batch_size
        )
        self.pattern_filter = PatternFilter(
            ignore_patterns=watchdog_config.ignore_patterns,
            ignore_directories=watchdog_config.ignore_directories,
        )
        
        # Event handler
        self.event_handler = FileEventHandler(
            processor=processor,
            pattern_filter=self.pattern_filter,
            debouncer=self.debouncer
        )

        self.watch_dirs = self._get_watch_directories()
        
        # State
        self.is_running = False
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'files_processed': 0,
            'errors': 0,
            'last_event': None,
            'watched_directories': len(self.watch_dirs)
        }
        
        # Callbacks
        self.callbacks = {
            'on_file_created': [],
            'on_file_modified': [],
            'on_file_deleted': [],
            'on_file_moved': [],
            'on_batch_complete': []
        }
        
        logger.info(f"FileMonitor initialized with {len(self.watch_dirs)} directories to watch")
    
    def _get_watch_directories(self) -> List[Path]:
        watch_dirs = []

        # main source directory
        source_dir = self.config.paths.source
        if source_dir and source_dir.exists() and source_dir.is_dir():
            watch_dirs.append(source_dir)

        # additional
        additional_dirs = getattr(self.config.watchdog, "watch_directories", [])
        for dir_path in additional_dirs:
            p = Path(dir_path)
            if p.exists() and p.is_dir():
                watch_dirs.append(p)

        # remove duplicates
        unique = []
        seen = set()
        for d in watch_dirs:
            s = str(d)
            if s not in seen:
                seen.add(s)
                unique.append(d)
        return unique

    
    async def start(self) -> bool:
        """Start monitoring directories"""
        if self.is_running:
            logger.warning("FileMonitor is already running")
            return True
        
        if not self.watch_dirs:
            logger.error("No directories to watch")
            return False
        
        try:
            # Initialize watchdog observer
            self.observer = Observer()
            
            # Schedule watches for all directories
            for watch_dir in self.watch_dirs:
                try:
                    self.observer.schedule(
                        self.event_handler,
                        str(watch_dir),
                        recursive=self.recursive
                    )
                    logger.info(f"Watching directory: {watch_dir} (recursive: {self.recursive})")
                except Exception as e:
                    logger.error(f"Failed to watch directory {watch_dir}: {e}")
            
            if not self.observer._watches:
                logger.error("No directories could be watched")
                return False
            
            # Start observer
            self.observer.start()
            self.is_running = True
            
            # Start debouncer processing loop
            asyncio.create_task(self._process_debounced_events())
            
            logger.info("FileMonitor started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start FileMonitor: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop monitoring directories"""
        if not self.is_running:
            return True
        
        try:
            # Stop observer
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=10)
            
            # Stop debouncer
            if self.debouncer:
                await self.debouncer.stop()
            
            self.is_running = False
            logger.info("FileMonitor stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping FileMonitor: {e}")
            return False
    
    async def _process_debounced_events(self):
        """Process debounced events in background"""
        logger.info("Starting debounced event processor")
        
        while self.is_running:
            try:
                # Get batched events from debouncer
                batched_events = await self.debouncer.get_batch()
                
                if batched_events:
                    logger.info(f"Processing batch of {len(batched_events)} events")
                    
                    # Process each event
                    for event in batched_events:
                        await self._process_event(event)
                    
                    # Update statistics
                    self.stats['events_processed'] += len(batched_events)
                    
                    # Call batch complete callbacks
                    await self._call_callbacks('on_batch_complete', batched_events)
                
                # Small sleep to prevent CPU spinning
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in debounced event processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_event(self, event: WatchdogEvent):
        """Process a single watchdog event"""
        try:
            self.stats['last_event'] = datetime.now()
            
            # Check if file should be ignored
            if self.pattern_filter.should_ignore(event.src_path):
                logger.debug(f"Ignoring event for {event.src_path} (pattern filter)")
                return
            
            # Update statistics
            self.stats['events_received'] += 1
            
            # Process based on event type
            if event.event_type == EventType.CREATED:
                await self._handle_file_created(event)
            elif event.event_type == EventType.MODIFIED:
                await self._handle_file_modified(event)
            elif event.event_type == EventType.MOVED:
                await self._handle_file_moved(event)
            elif event.event_type == EventType.DELETED:
                await self._handle_file_deleted(event)
            
            # Call type-specific callbacks
            callback_key = f"on_file_{event.event_type.value}"
            await self._call_callbacks(callback_key, event)
            
        except Exception as e:
            logger.error(f"Error processing event {event}: {e}")
            self.stats['errors'] += 1
    
    async def _handle_file_created(self, event: WatchdogEvent):
        """Handle file creation event"""
        logger.info(f"File created: {event.src_path}")
        
        # Wait for file to stabilize (in case it's still being written)
        if not await self._wait_for_file_stable(event.src_path):
            logger.warning(f"File {event.src_path} did not stabilize, skipping")
            return
        
        # Process the file if we have a processor
        if self.processor:
            try:
                await self.processor.process_single_file(event.src_path)
                self.stats['files_processed'] += 1
            except Exception as e:
                logger.error(f"Error processing created file {event.src_path}: {e}")
    
    async def _handle_file_modified(self, event: WatchdogEvent):
        """Handle file modification event"""
        logger.debug(f"File modified: {event.src_path}")
        
        # For modifications, we might want to re-process the file
        # This can be configured based on file type
        if self._should_reprocess_on_modify(event.src_path):
            logger.info(f"Re-processing modified file: {event.src_path}")
            
            if self.processor:
                try:
                    await self.processor.process_single_file(event.src_path)
                    self.stats['files_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing modified file {event.src_path}: {e}")
    
    async def _handle_file_moved(self, event: WatchdogEvent):
        """Handle file move/rename event"""
        logger.info(f"File moved: {event.src_path} -> {event.dest_path}")
        
        # Check if destination should be ignored
        if event.dest_path and self.pattern_filter.should_ignore(event.dest_path):
            logger.debug(f"Ignoring moved file destination {event.dest_path}")
            return
        
        # Process the destination file if it's a creation
        if event.dest_path and event.dest_path.exists():
            if not await self._wait_for_file_stable(event.dest_path):
                return
            
            if self.processor:
                try:
                    await self.processor.process_single_file(event.dest_path)
                    self.stats['files_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing moved file {event.dest_path}: {e}")
    
    async def _handle_file_deleted(self, event: WatchdogEvent):
        """Handle file deletion event"""
        logger.info(f"File deleted: {event.src_path}")
        
        # Update database or perform cleanup
        # This would typically mark the file as deleted in the database
        
        # Call any cleanup handlers
        await self._call_callbacks('on_file_deleted', event)
    
    async def _wait_for_file_stable(self, file_path: Path, 
                                   max_wait: float = 30.0, 
                                   check_interval: float = 0.5) -> bool:
        """
        Wait for file to stabilize (stop changing size)
        
        Args:
            file_path: Path to file
            max_wait: Maximum wait time in seconds
            check_interval: Interval between checks
            
        Returns:
            True if file stabilized, False if timeout
        """
        if not file_path.exists():
            return False
        
        start_time = time.time()
        last_size = -1
        stable_count = 0
        required_stable = 3  # Need 3 consecutive stable checks
        
        while (time.time() - start_time) < max_wait:
            try:
                current_size = file_path.stat().st_size
                
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= required_stable:
                        return True
                else:
                    stable_count = 0
                    last_size = current_size
                
                await asyncio.sleep(check_interval)
                
            except (FileNotFoundError, PermissionError):
                # File disappeared or inaccessible
                return False
        
        logger.warning(f"File {file_path} did not stabilize within {max_wait} seconds")
        return False
    
    def _should_reprocess_on_modify(self, file_path: Path) -> bool:
        """Determine if file should be re-processed on modification"""
        # Configuration: which file types to re-process on modify
        reprocess_types = getattr(self.config.watchdog, "reprocess_on_modify", [])
        
        if not reprocess_types:
            return False
        
        # Check file extension
        file_ext = file_path.suffix.lower()
        
        # Check if this file type should be re-processed
        for pattern in reprocess_types:
            if pattern.startswith('.'):
                # Exact extension match
                if file_ext == pattern.lower():
                    return True
            elif pattern == '*':
                # All files
                return True
            elif pattern.endswith('*'):
                # Wildcard match (e.g., '*.raw')
                if file_ext.endswith(pattern[1:]):
                    return True
        
        return False
    
    def register_callback(self, event_type: str, callback: Callable):
        """
        Register a callback for specific event type
        
        Args:
            event_type: Event type (e.g., 'on_file_created')
            callback: Callback function
        """
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.debug(f"Registered callback for {event_type}")
        else:
            logger.warning(f"Unknown event type for callback: {event_type}")
    
    async def _call_callbacks(self, event_type: str, *args, **kwargs):
        """Call all registered callbacks for event type"""
        if event_type not in self.callbacks:
            return
        
        for callback in self.callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args, **kwargs)
                else:
                    callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in callback for {event_type}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitor status"""
        return {
            'is_running': self.is_running,
            'watched_directories': [str(d) for d in self.watch_dirs],
            'recursive': self.recursive,
            'debounce_time': self.debounce_time,
            'stats': self.stats.copy(),
            'pattern_filter': {
                'ignore_patterns': self.pattern_filter.ignore_patterns,
                'ignore_directories': self.pattern_filter.ignore_directories
            }
        }
    
    async def force_scan(self, directory: Optional[Path] = None) -> Dict[str, Any]:
        """
        Force scan of directory for new files
        
        Args:
            directory: Directory to scan (None for all watched directories)
            
        Returns:
            Scan results
        """
        scan_dirs = [directory] if directory else self.watch_dirs
        
        results = {
            'scanned_directories': [],
            'files_found': 0,
            'files_processed': 0,
            'errors': 0
        }
        
        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                logger.warning(f"Scan directory does not exist: {scan_dir}")
                continue
            
            results['scanned_directories'].append(str(scan_dir))
            
            try:
                # Find all files
                file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.heic', '*.mp4', '*.mov']
                files = []
                
                for pattern in file_patterns:
                    if self.recursive:
                        files.extend(scan_dir.rglob(pattern))
                    else:
                        files.extend(scan_dir.glob(pattern))
                
                # Filter out ignored files
                filtered_files = []
                for file_path in files:
                    if not self.pattern_filter.should_ignore(file_path):
                        filtered_files.append(file_path)
                
                results['files_found'] += len(filtered_files)
                
                # Process files
                if self.processor:
                    for file_path in filtered_files:
                        try:
                            await self.processor.process_single_file(file_path)
                            results['files_processed'] += 1
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
                            results['errors'] += 1
                
                logger.info(f"Force scan completed for {scan_dir}: {len(filtered_files)} files")
                
            except Exception as e:
                logger.error(f"Error scanning directory {scan_dir}: {e}")
                results['errors'] += 1
        
        return results
    
    def add_watch_directory(self, directory: Path) -> bool:
        """
        Add a new directory to watch
        
        Args:
            directory: Directory to add
            
        Returns:
            True if successful, False otherwise
        """
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Cannot watch non-existent directory: {directory}")
            return False
        
        if directory in self.watch_dirs:
            logger.warning(f"Directory already being watched: {directory}")
            return True
        
        try:
            # Add to observer if running
            if self.is_running and self.observer:
                self.observer.schedule(
                    self.event_handler,
                    str(directory),
                    recursive=self.recursive
                )
            
            # Add to internal list
            self.watch_dirs.append(directory)
            self.stats['watched_directories'] = len(self.watch_dirs)
            
            logger.info(f"Added watch directory: {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add watch directory {directory}: {e}")
            return False
    
    def remove_watch_directory(self, directory: Path) -> bool:
        """
        Remove a directory from watch list
        
        Args:
            directory: Directory to remove
            
        Returns:
            True if successful, False otherwise
        """
        if directory not in self.watch_dirs:
            logger.warning(f"Directory not being watched: {directory}")
            return False
        
        try:
            # Remove from observer if running
            # Note: watchdog doesn't have a direct way to unschedule
            # We would need to stop and restart the observer
            
            # Remove from internal list
            self.watch_dirs.remove(directory)
            self.stats['watched_directories'] = len(self.watch_dirs)
            
            logger.info(f"Removed watch directory: {directory}")
            
            # If running, we need to restart the observer
            if self.is_running:
                asyncio.create_task(self._restart_observer())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove watch directory {directory}: {e}")
            return False
    
    async def _restart_observer(self):
        """Restart observer (e.g., after directory changes)"""
        logger.info("Restarting observer due to directory changes")
        
        await self.stop()
        await asyncio.sleep(1)
        await self.start()