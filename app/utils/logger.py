"""
Logging configuration for Mnemosyne
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

# Custom formatter for JSON logs
class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        # Add thread/process info
        if record.process:
            log_record['process_id'] = record.process
        if record.thread:
            log_record['thread_id'] = record.thread
        
        return json.dumps(log_record, default=str)


class ColorFormatter(logging.Formatter):
    """Color formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red background
        'RESET': '\033[0m',       # Reset
    }
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        # Colorize the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            levelname_color = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
            record.levelname = levelname_color
        
        # Colorize the message for ERROR and CRITICAL
        if record.levelno >= logging.ERROR:
            record.msg = f"{self.COLORS['ERROR']}{record.msg}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "text",  # text, json, or color
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, only console logging)
        log_format: Format of logs (text, json, or color)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Convert string level to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter based on format type
    if log_format.lower() == "json":
        console_formatter = JsonFormatter()
    elif log_format.lower() == "color":
        console_formatter = ColorFormatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:  # text
        console_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use RotatingFileHandler for log rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        # Use JSON formatter for file if requested, otherwise text
        if log_format.lower() == "json":
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
        
        # Log file location
        root_logger.info(f"Logging to file: {log_path}")
    
    # Set levels for specific loggers
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('sqlalchemy.pool').setLevel(logging.WARNING)
    logging.getLogger('watchdog').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    
    # Log startup message
    root_logger.info(f"Logging configured. Level: {log_level}, Format: {log_format}")
    
    return root_logger


def get_logger(name: str = None, extra: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Get logger with optional extra fields for structured logging
    
    Args:
        name: Logger name (usually __name__)
        extra: Extra fields to include in log records
    """
    logger = logging.getLogger(name)
    
    # Add extra fields to log records if provided
    if extra:
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra = extra
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    return logger


def log_exception(logger: logging.Logger, exception: Exception, 
                 message: str = "Exception occurred", extra: Optional[Dict] = None):
    """
    Log exception with traceback and optional extra context
    
    Args:
        logger: Logger instance
        exception: Exception to log
        message: Custom message
        extra: Extra context information
    """
    exc_info = (type(exception), exception, exception.__traceback__)
    
    if extra:
        logger.error(message, exc_info=exc_info, extra=extra)
    else:
        logger.error(message, exc_info=exc_info)


def setup_performance_logging(log_file: str = "./logs/performance.log"):
    """
    Setup separate logging for performance metrics
    
    Args:
        log_file: Path to performance log file
    """
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False  # Don't propagate to root logger
    
    # Remove existing handlers
    for handler in perf_logger.handlers[:]:
        perf_logger.removeHandler(handler)
    
    # Create file handler for performance logs
    perf_path = Path(log_file)
    perf_path.parent.mkdir(parents=True, exist_ok=True)
    
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        filename=perf_path,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    
    # JSON formatter for performance logs
    formatter = JsonFormatter()
    file_handler.setFormatter(formatter)
    
    perf_logger.addHandler(file_handler)
    
    return perf_logger


class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None, 
                 extra: Optional[Dict] = None):
        self.operation = operation
        self.logger = logger or logging.getLogger('performance')
        self.extra = extra or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from datetime import datetime
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        log_data = {
            'operation': self.operation,
            'duration_seconds': duration,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            **self.extra
        }
        
        if exc_type:
            log_data['error'] = str(exc_val)
            self.logger.error(f"Performance measurement for {self.operation}", 
                            extra=log_data)
        else:
            self.logger.info(f"Performance measurement for {self.operation}", 
                           extra=log_data)


def log_performance(operation: str, logger: Optional[logging.Logger] = None,
                   extra: Optional[Dict] = None):
    """Decorator for logging function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceLogger(operation, logger, extra):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize default logging on import
_root_logger = None

def init_default_logging():
    """Initialize default logging configuration"""
    global _root_logger
    if _root_logger is None:
        _root_logger = setup_logging()
    return _root_logger


# Auto-initialize on import
init_default_logging()