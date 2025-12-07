from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


class EventType(Enum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    MOVED = "moved"
    CLOSED = "closed"


@dataclass
class WatchdogEvent:
    event_type: EventType
    src_path: Path
    dest_path: Optional[Path] = None
    is_directory: bool = False
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def __str__(self):
        if self.dest_path:
            return f"{self.event_type.value}: {self.src_path} -> {self.dest_path}"
        return f"{self.event_type.value}: {self.src_path}"
