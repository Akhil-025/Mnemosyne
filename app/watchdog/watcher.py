# app/watchdog/watcher.py

"""
Directory watcher implementations
"""
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Set, Callable
from datetime import datetime
import time

from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver

from .monitor import WatchdogEvent, EventType
from .handlers import EventHandler
from .patterns import PatternFilter
from .events import EventType, WatchdogEvent

logger = logging.getLogger(__name__)


class DirectoryWatcher:
    """
    Directory watcher with polling support
    """
    
    def __init__(self, directory: Path,
                 handler: EventHandler,
                 recursive: bool = True,
                 use_polling: bool = False,
                 poll_interval: float = 1.0):
        """
        Initialize directory watcher
        
        Args:
            directory: Directory to watch
            handler: Event handler
            recursive: Watch recursively
            use_polling: Use polling instead of OS events
            poll_interval: Polling interval in seconds
        """
        self.directory = directory
        self.handler = handler
        self.recursive = recursive
        self.use_polling = use_polling
        self.poll_interval = poll_interval
        
        # Observer instance
        self.observer = None
        
        # State
        self.is_watching = False
        self.stats = {
            'start_time': None,
            'total_events': 0,
            'last_event': None,
        }
        
        logger.info(f"DirectoryWatcher initialized for {directory}")
    
    def start(self) -> bool:
        """Start watching directory"""
        if self.is_watching:
            logger.warning(f"Already watching directory: {self.directory}")
            return True
        
        if not self.directory.exists():
            logger.error(f"Directory does not exist: {self.directory}")
            return False
        
        try:
            # Create appropriate observer
            if self.use_polling:
                self.observer = PollingObserver(timeout=self.poll_interval)
                logger.debug(f"Using polling observer (interval: {self.poll_interval}s)")
            else:
                self.observer = Observer()
                logger.debug("Using OS event observer")
            
            # Schedule watch
            self.observer.schedule(
                self.handler,
                str(self.directory),
                recursive=self.recursive
            )
            
            # Start observer
            self.observer.start()
            self.is_watching = True
            self.stats['start_time'] = datetime.now()
            
            logger.info(f"Started watching directory: {self.directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start watching {self.directory}: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop watching directory"""
        if not self.is_watching:
            return True
        
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=10)
                self.observer = None
            
            self.is_watching = False
            logger.info(f"Stopped watching directory: {self.directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping watcher for {self.directory}: {e}")
            return False
    
    def restart(self) -> bool:
        """Restart directory watcher"""
        logger.info(f"Restarting watcher for {self.directory}")
        
        self.stop()
        time.sleep(0.5)  # Brief pause
        return self.start()
    
    def get_status(self) -> Dict[str, Any]:
        """Get watcher status"""
        handler_stats = self.handler.get_stats() if hasattr(self.handler, 'get_stats') else {}
        
        return {
            'directory': str(self.directory),
            'is_watching': self.is_watching,
            'recursive': self.recursive,
            'use_polling': self.use_polling,
            'poll_interval': self.poll_interval if self.use_polling else None,
            'start_time': self.stats['start_time'],
            'duration': (datetime.now() - self.stats['start_time']).total_seconds() 
                       if self.stats['start_time'] else 0,
            'stats': {**self.stats, **handler_stats},
        }
    
    def update_handler(self, handler: EventHandler) -> bool:
        """
        Update event handler
        
        Args:
            handler: New event handler
            
        Returns:
            True if successful
        """
        if self.is_watching:
            # Need to restart to apply new handler
            was_watching = True
            self.stop()
        else:
            was_watching = False
        
        # Update handler
        self.handler = handler
        
        # Restart if was watching
        if was_watching:
            return self.start()
        
        return True


class RecursiveWatcher:
    """
    Watcher that manages multiple directory watchers recursively
    """
    
    def __init__(self, root_directory: Path,
                 handler_factory: Callable[[Path], EventHandler],
                 pattern_filter: Optional[PatternFilter] = None,
                 use_polling: bool = False,
                 poll_interval: float = 1.0):
        """
        Initialize recursive watcher
        
        Args:
            root_directory: Root directory to watch
            handler_factory: Factory function to create handlers for subdirectories
            pattern_filter: Pattern filter for ignoring directories
            use_polling: Use polling observers
            poll_interval: Polling interval
        """
        self.root_directory = root_directory
        self.handler_factory = handler_factory
        self.pattern_filter = pattern_filter or PatternFilter()
        self.use_polling = use_polling
        self.poll_interval = poll_interval
        
        # Watchers by directory
        self.watchers: Dict[Path, DirectoryWatcher] = {}
        
        # State
        self.is_running = False
        self.scan_task = None
        
        # Configuration
        self.scan_interval = 300  # Scan for new directories every 5 minutes
        self.max_depth = 10  # Maximum recursion depth
        
        logger.info(f"RecursiveWatcher initialized for {root_directory}")
    
    async def start(self) -> bool:
        """Start recursive watching"""
        if self.is_running:
            return True
        
        if not self.root_directory.exists():
            logger.error(f"Root directory does not exist: {self.root_directory}")
            return False
        
        try:
            # Start with root directory
            self._start_watcher(self.root_directory)
            
            # Start periodic scan for new directories
            self.is_running = True
            self.scan_task = asyncio.create_task(self._periodic_scan())
            
            logger.info(f"RecursiveWatcher started for {self.root_directory}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RecursiveWatcher: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop recursive watching"""
        if not self.is_running:
            return True
        
        try:
            # Stop scan task
            self.is_running = False
            if self.scan_task:
                self.scan_task.cancel()
                try:
                    await self.scan_task
                except asyncio.CancelledError:
                    pass
            
            # Stop all watchers
            for watcher in list(self.watchers.values()):
                watcher.stop()
            
            self.watchers.clear()
            
            logger.info("RecursiveWatcher stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping RecursiveWatcher: {e}")
            return False
    
    def _start_watcher(self, directory: Path) -> bool:
        """Start watcher for specific directory"""
        # Check if already watching
        if directory in self.watchers:
            return True
        
        # Check if directory should be ignored
        if self.pattern_filter.should_ignore(directory):
            logger.debug(f"Skipping ignored directory: {directory}")
            return False
        
        # Check depth
        depth = len(directory.relative_to(self.root_directory).parts)
        if depth > self.max_depth:
            logger.debug(f"Skipping directory at excessive depth: {directory}")
            return False
        
        try:
            # Create handler for this directory
            handler = self.handler_factory(directory)
            
            # Create and start watcher
            watcher = DirectoryWatcher(
                directory=directory,
                handler=handler,
                recursive=False,  # We handle recursion manually
                use_polling=self.use_polling,
                poll_interval=self.poll_interval
            )
            
            if watcher.start():
                self.watchers[directory] = watcher
                logger.info(f"Started watcher for directory: {directory}")
                return True
            else:
                logger.warning(f"Failed to start watcher for {directory}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting watcher for {directory}: {e}")
            return False
    
    def _stop_watcher(self, directory: Path) -> bool:
        """Stop watcher for specific directory"""
        if directory not in self.watchers:
            return True
        
        try:
            watcher = self.watchers.pop(directory)
            watcher.stop()
            logger.info(f"Stopped watcher for directory: {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping watcher for {directory}: {e}")
            return False
    
    async def _periodic_scan(self):
        """Periodically scan for new directories"""
        logger.info("Starting periodic directory scan")
        
        while self.is_running:
            try:
                await self._scan_directories()
                await asyncio.sleep(self.scan_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic scan: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    async def _scan_directories(self):
        """Scan for new and removed directories"""
        # Find all directories
        all_directories = set()
        
        try:
            # Walk through directory tree
            for dir_path in self.root_directory.rglob('*'):
                if dir_path.is_dir():
                    all_directories.add(dir_path)
        except Exception as e:
            logger.error(f"Error scanning directories: {e}")
            return
        
        # Add root directory
        all_directories.add(self.root_directory)
        
        # Start watchers for new directories
        for directory in all_directories:
            if directory not in self.watchers:
                self._start_watcher(directory)
        
        # Stop watchers for removed directories
        for directory in list(self.watchers.keys()):
            if directory not in all_directories:
                self._stop_watcher(directory)
        
        logger.debug(f"Directory scan complete: {len(self.watchers)} active watchers")
    
    def force_rescan(self):
        """Force immediate rescan of directories"""
        if self.is_running:
            asyncio.create_task(self._scan_directories())
    
    def get_watcher_status(self, directory: Optional[Path] = None) -> Dict[str, Any]:
        """Get status of watcher(s)"""
        if directory:
            if directory in self.watchers:
                return self.watchers[directory].get_status()
            else:
                return {'error': f'Not watching directory: {directory}'}
        else:
            # Return status for all watchers
            status = {
                'root_directory': str(self.root_directory),
                'is_running': self.is_running,
                'total_watchers': len(self.watchers),
                'watchers': {},
            }
            
            for dir_path, watcher in self.watchers.items():
                status['watchers'][str(dir_path)] = watcher.get_status()
            
            return status
    
    def add_exclusion(self, pattern: str):
        """Add directory exclusion pattern"""
        self.pattern_filter.add_pattern(pattern, is_directory=True)
        
        # Check if any currently watched directories should be excluded
        for directory in list(self.watchers.keys()):
            if self.pattern_filter.should_ignore(directory):
                self._stop_watcher(directory)
    
    def remove_exclusion(self, pattern: str):
        """Remove directory exclusion pattern"""
        self.pattern_filter.remove_pattern(pattern)
        
        # Rescan to potentially add back excluded directories
        self.force_rescan()
    
    async def watch_single_file(self, file_path: Path) -> bool:
        """
        Watch a single file specifically
        
        Args:
            file_path: File to watch
            
        Returns:
            True if watching started
        """
        directory = file_path.parent
        
        # Ensure directory is being watched
        if directory not in self.watchers:
            self._start_watcher(directory)
        
        # Check if file is already being monitored
        # This would require additional logic in the handler
        
        return directory in self.watchers


class SmartWatcher:
    """
    Smart watcher that adapts based on system load and activity
    """
    
    def __init__(self, config: Dict):
        """
        Initialize smart watcher
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Adaptive parameters
        self.poll_interval = config.get('poll_interval', 1.0)
        self.max_poll_interval = config.get('max_poll_interval', 10.0)
        self.min_poll_interval = config.get('min_poll_interval', 0.1)
        
        # Load thresholds
        self.cpu_threshold = config.get('cpu_threshold', 80.0)  # percent
        self.memory_threshold = config.get('memory_threshold', 90.0)  # percent
        self.event_rate_threshold = config.get('event_rate_threshold', 100)  # events/second
        
        # Components
        self.watcher = None
        self.adaptive_task = None
        
        # Monitoring state
        self.is_running = False
        self.current_load = {
            'cpu': 0.0,
            'memory': 0.0,
            'event_rate': 0.0,
        }
        
        # Statistics
        self.adaptation_stats = {
            'interval_changes': 0,
            'last_interval_change': None,
            'high_load_periods': 0,
        }
        
        logger.info("SmartWatcher initialized")
    
    async def start(self, root_directory: Path, handler_factory: Callable) -> bool:
        """Start smart watcher"""
        if self.is_running:
            return True
        
        try:
            # Create recursive watcher
            self.watcher = RecursiveWatcher(
                root_directory=root_directory,
                handler_factory=handler_factory,
                use_polling=True,  # Always use polling for adaptive control
                poll_interval=self.poll_interval
            )
            
            # Start watcher
            await self.watcher.start()
            
            # Start adaptive control task
            self.is_running = True
            self.adaptive_task = asyncio.create_task(self._adaptive_control_loop())
            
            logger.info("SmartWatcher started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start SmartWatcher: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop smart watcher"""
        if not self.is_running:
            return True
        
        try:
            self.is_running = False
            
            # Stop adaptive task
            if self.adaptive_task:
                self.adaptive_task.cancel()
                try:
                    await self.adaptive_task
                except asyncio.CancelledError:
                    pass
            
            # Stop watcher
            if self.watcher:
                await self.watcher.stop()
            
            logger.info("SmartWatcher stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping SmartWatcher: {e}")
            return False
    
    async def _adaptive_control_loop(self):
        """Adaptive control loop"""
        check_interval = 10.0  # Check every 10 seconds
        
        while self.is_running:
            try:
                # Get current system load
                await self._update_system_load()
                
                # Adjust polling interval based on load
                await self._adjust_polling_interval()
                
                # Log status periodically
                await self._log_status()
                
                await asyncio.sleep(check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptive control loop: {e}")
                await asyncio.sleep(check_interval)
    
    async def _update_system_load(self):
        """Update current system load metrics"""
        try:
            # Get CPU usage
            import psutil
            self.current_load['cpu'] = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            self.current_load['memory'] = memory.percent
            
            # Calculate event rate (would need to track events)
            # For now, use a placeholder
            self.current_load['event_rate'] = 0.0
            
        except ImportError:
            logger.warning("psutil not installed, system load monitoring disabled")
            self.current_load = {'cpu': 0.0, 'memory': 0.0, 'event_rate': 0.0}
        except Exception as e:
            logger.error(f"Error updating system load: {e}")
    
    async def _adjust_polling_interval(self):
        """Adjust polling interval based on system load"""
        old_interval = self.poll_interval
        
        # Check if system is under high load
        high_load = (
            self.current_load['cpu'] > self.cpu_threshold or
            self.current_load['memory'] > self.memory_threshold
        )
        
        if high_load:
            # Increase interval to reduce load
            self.poll_interval = min(
                self.poll_interval * 1.5,
                self.max_poll_interval
            )
            self.adaptation_stats['high_load_periods'] += 1
        else:
            # System has capacity, can reduce interval
            self.poll_interval = max(
                self.poll_interval * 0.9,
                self.min_poll_interval
            )
        
        # Check if interval changed significantly
        if abs(self.poll_interval - old_interval) > 0.1:
            self.adaptation_stats['interval_changes'] += 1
            self.adaptation_stats['last_interval_change'] = datetime.now()
            
            # Update watcher if it exists
            if self.watcher:
                # Would need to restart watchers with new interval
                logger.info(f"Adapting poll interval: {old_interval:.2f}s -> {self.poll_interval:.2f}s")
    
    async def _log_status(self):
        """Log current status"""
        logger.debug(
            f"SmartWatcher status: "
            f"CPU={self.current_load['cpu']:.1f}%, "
            f"Memory={self.current_load['memory']:.1f}%, "
            f"PollInterval={self.poll_interval:.2f}s"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get smart watcher status"""
        watcher_status = self.watcher.get_watcher_status() if self.watcher else {}
        
        return {
            'is_running': self.is_running,
            'adaptive_parameters': {
                'poll_interval': self.poll_interval,
                'min_poll_interval': self.min_poll_interval,
                'max_poll_interval': self.max_poll_interval,
                'cpu_threshold': self.cpu_threshold,
                'memory_threshold': self.memory_threshold,
            },
            'current_load': self.current_load,
            'adaptation_stats': self.adaptation_stats,
            'watcher': watcher_status,
        }