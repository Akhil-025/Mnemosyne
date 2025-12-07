# app/watchdog/debounce.py

"""
Event debouncing for file system events
"""
import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import hashlib

from .events import WatchdogEvent, EventType

logger = logging.getLogger(__name__)


@dataclass
class DebouncedEvent:
    """Event with debouncing metadata"""
    event: WatchdogEvent
    first_seen: datetime
    last_seen: datetime
    count: int = 1
    processed: bool = False
    
    def update(self):
        """Update event timestamp and count"""
        self.last_seen = datetime.now()
        self.count += 1
    
    def age(self) -> float:
        """Get age in seconds"""
        return (datetime.now() - self.first_seen).total_seconds()
    
    def is_stale(self, timeout: float) -> bool:
        """Check if event is stale (older than timeout)"""
        return self.age() > timeout


class Debouncer:
    """
    Base debouncer for grouping similar events
    """
    
    def __init__(self, debounce_time: float = 2.0):
        """
        Initialize debouncer
        
        Args:
            debounce_time: Time in seconds to wait before processing events
        """
        self.debounce_time = debounce_time
        self.events: Dict[str, DebouncedEvent] = {}
        self.lock = asyncio.Lock()
        
        # Statistics
        self.stats = {
            'total_events': 0,
            'debounced_events': 0,
            'processed_events': 0,
            'dropped_events': 0,
        }
    
    def _generate_event_key(self, event: WatchdogEvent) -> str:
        """
        Generate unique key for event grouping
        
        Args:
            event: Watchdog event
            
        Returns:
            Unique key string
        """
        # Base key on event type and source path
        key_parts = [
            event.event_type.value,
            str(event.src_path).lower(),
        ]
        
        # For move events, include destination
        if event.dest_path:
            key_parts.append(str(event.dest_path).lower())
        
        # Create hash for consistent key
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def add_event(self, event: WatchdogEvent) -> Optional[DebouncedEvent]:
        """
        Add event to debouncer
        
        Args:
            event: Watchdog event
            
        Returns:
            Debounced event if ready for processing, None otherwise
        """
        async with self.lock:
            self.stats['total_events'] += 1
            
            # Generate key
            event_key = self._generate_event_key(event)
            
            # Check if event already exists
            if event_key in self.events:
                # Update existing event
                debounced_event = self.events[event_key]
                debounced_event.update()
                self.stats['debounced_events'] += 1
                
                logger.debug(f"Debounced event {event_key} (count: {debounced_event.count})")
            else:
                # Create new debounced event
                now = datetime.now()
                debounced_event = DebouncedEvent(
                    event=event,
                    first_seen=now,
                    last_seen=now
                )
                self.events[event_key] = debounced_event
            
            # Check if event is ready for processing
            time_since_last = (datetime.now() - debounced_event.last_seen).total_seconds()
            if time_since_last >= self.debounce_time:
                # Mark as processed
                debounced_event.processed = True
                self.stats['processed_events'] += 1
                
                # Remove from active events
                del self.events[event_key]
                
                return debounced_event
            
            return None
    
    async def get_ready_events(self) -> List[DebouncedEvent]:
        """
        Get all events that are ready for processing
        
        Returns:
            List of debounced events ready for processing
        """
        async with self.lock:
            ready_events = []
            now = datetime.now()
            
            # Find events that are ready
            event_keys = list(self.events.keys())
            for event_key in event_keys:
                debounced_event = self.events[event_key]
                
                # Check if event has expired (ready for processing)
                time_since_last = (now - debounced_event.last_seen).total_seconds()
                if time_since_last >= self.debounce_time:
                    # Mark as processed
                    debounced_event.processed = True
                    ready_events.append(debounced_event)
                    
                    # Remove from active events
                    del self.events[event_key]
            
            self.stats['processed_events'] += len(ready_events)
            return ready_events
    
    async def cleanup_stale_events(self, max_age: float = 300.0) -> int:
        """
        Clean up stale events that haven't been processed
        
        Args:
            max_age: Maximum age in seconds before cleanup
            
        Returns:
            Number of events cleaned up
        """
        async with self.lock:
            stale_keys = []
            now = datetime.now()
            
            # Find stale events
            for event_key, debounced_event in self.events.items():
                age = (now - debounced_event.last_seen).total_seconds()
                if age > max_age:
                    stale_keys.append(event_key)
            
            # Remove stale events
            for event_key in stale_keys:
                del self.events[event_key]
            
            self.stats['dropped_events'] += len(stale_keys)
            
            if stale_keys:
                logger.info(f"Cleaned up {len(stale_keys)} stale events")
            
            return len(stale_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get debouncer statistics"""
        return {
            **self.stats,
            'active_events': len(self.events),
            'debounce_time': self.debounce_time,
        }


class EventDebouncer:
    """
    Advanced debouncer with batch processing and prioritization
    """
    
    def __init__(self, debounce_time: float = 2.0, 
                 max_events: int = 10000,
                 batch_size: int = 100):
        """
        Initialize event debouncer
        
        Args:
            debounce_time: Time in seconds to wait before processing
            max_events: Maximum number of events to buffer
            batch_size: Maximum batch size for processing
        """
        self.debounce_time = debounce_time
        self.max_events = max_events
        self.batch_size = batch_size
        
        # Multiple debouncers for different event types
        self.debouncers = {
            'created': Debouncer(debounce_time),
            'modified': Debouncer(debounce_time * 0.5),  # Shorter debounce for modifications
            'moved': Debouncer(debounce_time),
            'deleted': Debouncer(debounce_time * 2.0),   # Longer debounce for deletions
        }
        
        # Event queues
        self.event_queue = asyncio.Queue(maxsize=max_events)
        self.batch_queue = asyncio.Queue()
        
        # Control flags
        self.is_running = False
        self.processing_task = None
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_debounced': 0,
            'batches_processed': 0,
            'queue_size': 0,
            'dropped_events': 0,
        }
    
    async def start(self):
        """Start debouncer processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_task = asyncio.create_task(self._process_events())
        
        # Start cleanup task
        asyncio.create_task(self._periodic_cleanup())
        
        logger.info(f"EventDebouncer started (debounce_time={self.debounce_time}s)")
    
    async def stop(self):
        """Stop debouncer processing"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel processing task
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Process any remaining events
        await self._process_remaining_events()
        
        logger.info("EventDebouncer stopped")
    
    async def add_event(self, event: WatchdogEvent) -> bool:
        """
        Add event to debouncer
        
        Args:
            event: Watchdog event
            
        Returns:
            True if event was added, False if dropped
        """
        self.stats['events_received'] += 1
        
        # Check if queue is full
        if self.event_queue.full():
            self.stats['dropped_events'] += 1
            logger.warning(f"Event queue full, dropping event: {event}")
            return False
        
        try:
            # Add to queue with timeout
            await asyncio.wait_for(
                self.event_queue.put(event),
                timeout=0.1
            )
            
            self.stats['queue_size'] = self.event_queue.qsize()
            return True
            
        except asyncio.TimeoutError:
            self.stats['dropped_events'] += 1
            logger.warning(f"Timeout adding event to queue: {event}")
            return False
        except Exception as e:
            logger.error(f"Error adding event to queue: {e}")
            return False
    
    async def get_batch(self, timeout: Optional[float] = None) -> List[WatchdogEvent]:
        """
        Get a batch of processed events
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            List of watchdog events
        """
        try:
            if timeout:
                batch = await asyncio.wait_for(
                    self.batch_queue.get(),
                    timeout=timeout
                )
            else:
                batch = await self.batch_queue.get()
            
            return batch
            
        except asyncio.TimeoutError:
            return []
        except Exception as e:
            logger.error(f"Error getting batch: {e}")
            return []
    
    async def _process_events(self):
        """Main event processing loop"""
        logger.info("Starting event processing loop")
        
        while self.is_running:
            try:
                # Get event from queue
                event = await self.event_queue.get()
                
                # Update queue stats
                self.stats['queue_size'] = self.event_queue.qsize()
                
                # Debounce based on event type
                debouncer_key = event.event_type.value
                if debouncer_key in self.debouncers:
                    debouncer = self.debouncers[debouncer_key]
                    debounced_event = await debouncer.add_event(event)
                    
                    if debounced_event:
                        self.stats['events_debounced'] += 1
                        await self._handle_debounced_event(debounced_event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_debounced_event(self, debounced_event: DebouncedEvent):
        """Handle a debounced event"""
        # For now, just forward the original event
        # In a more advanced implementation, you might:
        # - Merge multiple events for the same file
        # - Prioritize events
        # - Apply rate limiting
        
        # Create batch if needed
        if self.batch_queue.empty():
            # Start new batch
            batch = [debounced_event.event]
            
            # Try to get more events quickly
            try:
                for _ in range(self.batch_size - 1):
                    # Non-blocking check for more events
                    try:
                        event = self.event_queue.get_nowait()
                        debouncer_key = event.event_type.value
                        if debouncer_key in self.debouncers:
                            debouncer = self.debouncers[debouncer_key]
                            debounced = await debouncer.add_event(event)
                            if debounced:
                                batch.append(debounced.event)
                                self.event_queue.task_done()
                    except asyncio.QueueEmpty:
                        break
            except Exception as e:
                logger.debug(f"Error batching events: {e}")
            
            # Add batch to output queue
            await self.batch_queue.put(batch)
            self.stats['batches_processed'] += 1
            
            logger.debug(f"Created batch of {len(batch)} events")
    
    async def _periodic_cleanup(self, interval: float = 60.0):
        """Periodic cleanup of stale events"""
        while self.is_running:
            try:
                await asyncio.sleep(interval)
                
                # Clean up stale events in all debouncers
                total_cleaned = 0
                for debouncer in self.debouncers.values():
                    cleaned = await debouncer.cleanup_stale_events()
                    total_cleaned += cleaned
                
                if total_cleaned > 0:
                    logger.debug(f"Periodic cleanup removed {total_cleaned} stale events")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def _process_remaining_events(self):
        """Process any remaining events when stopping"""
        logger.info("Processing remaining events before shutdown")
        
        # Process all events in queue
        processed = 0
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                
                # Add to appropriate debouncer
                debouncer_key = event.event_type.value
                if debouncer_key in self.debouncers:
                    debouncer = self.debouncers[debouncer_key]
                    debounced_event = await debouncer.add_event(event)
                    
                    if debounced_event:
                        # Create immediate batch for shutdown
                        batch = [debounced_event.event]
                        await self.batch_queue.put(batch)
                        processed += 1
                
                self.event_queue.task_done()
                
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                logger.error(f"Error processing remaining events: {e}")
        
        # Force process all debounced events
        for debouncer_name, debouncer in self.debouncers.items():
            ready_events = await debouncer.get_ready_events()
            if ready_events:
                batch = [de.event for de in ready_events]
                await self.batch_queue.put(batch)
                processed += len(batch)
        
        if processed > 0:
            logger.info(f"Processed {processed} remaining events before shutdown")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get debouncer statistics"""
        debouncer_stats = {}
        for name, debouncer in self.debouncers.items():
            debouncer_stats[name] = debouncer.get_stats()
        
        return {
            **self.stats,
            'debouncers': debouncer_stats,
            'is_running': self.is_running,
            'max_events': self.max_events,
            'batch_size': self.batch_size,
        }
    
    def get_queue_info(self) -> Dict[str, Any]:
        """Get queue information"""
        return {
            'event_queue_size': self.event_queue.qsize(),
            'event_queue_maxsize': self.event_queue.maxsize,
            'batch_queue_size': self.batch_queue.qsize(),
            'batch_queue_maxsize': self.batch_queue.maxsize,
        }
    
    async def flush(self) -> int:
        """
        Flush all pending events
        
        Returns:
            Number of events flushed
        """
        logger.info("Flushing all pending events")
        
        # Get all ready events from debouncers
        all_events = []
        for debouncer in self.debouncers.values():
            ready_events = await debouncer.get_ready_events()
            all_events.extend([de.event for de in ready_events])
        
        # Process events in queue
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get_nowait()
                all_events.append(event)
                self.event_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        # Create batches
        batch_size = self.batch_size
        for i in range(0, len(all_events), batch_size):
            batch = all_events[i:i + batch_size]
            await self.batch_queue.put(batch)
        
        logger.info(f"Flushed {len(all_events)} events")
        return len(all_events)