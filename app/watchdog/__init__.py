#app/watchdog/__init__.py

"""
Mnemosyne Watchdog Module
Digital Life Archival System - File System Monitoring
"""
from .monitor import FileMonitor
from .events import WatchdogEvent, EventType
from .debounce import Debouncer, EventDebouncer
from .patterns import PatternFilter, FilePatternMatcher
from .handlers import FileEventHandler, EventHandler, BatchEventHandler
from .watcher import DirectoryWatcher, RecursiveWatcher

__all__ = [
    'FileMonitor',
    'WatchdogEvent',
    'EventType',
    'Debouncer',
    'EventDebouncer',
    'PatternFilter',
    'FilePatternMatcher',
    'FileEventHandler',
    'EventHandler',
    'BatchEventHandler',
    'DirectoryWatcher',
    'RecursiveWatcher',
]