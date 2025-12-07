# app/utils/config.py

"""
Configuration management for Mnemosyne
"""
import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathConfig:
    """Path configuration"""
    source: Path = Path.home() / "Pictures"
    output: Path = Path.home() / "OrganizedPhotos"
    vault: Path = Path.home() / ".mnemosyne" / "vault"
    cache: Path = Path.home() / ".mnemosyne" / "cache"
    temp: Path = Path.home() / ".mnemosyne" / "temp"
    models: Path = Path.home() / ".mnemosyne" / "models"
    
    def __post_init__(self):
        # Convert strings to Path objects if needed
        for field_name in ['source', 'output', 'vault', 'cache', 'temp', 'models']:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))


@dataclass
class OllamaConfig:
    """Ollama configuration"""
    base_url: str = "http://localhost:11434"
    timeout: int = 300
    max_retries: int = 3
    image_model: str = "llava:13b"
    embedding_model: str = "nomic-embed-text:latest"
    text_model: str = "llama3:latest"
    enable_captioning: bool = True
    enable_embeddings: bool = True


@dataclass
class AIConfig:
    """AI/ML configuration"""
    enable_face_recognition: bool = True
    face_model: str = "buffalo_l"
    enable_ocr: bool = True
    ocr_model: str = "tesseract"  # or "easyocr"
    enable_aesthetic_scoring: bool = True
    enable_nsfw_detection: bool = False
    enable_generative_ai: bool = False
    
    # Performance
    use_gpu: bool = False
    gpu_memory_limit: Optional[int] = None
    cpu_threads: int = 4


@dataclass
class DatabaseConfig:
    """Database configuration"""
    url: str = "sqlite:///./data/mnemosyne.db"
    vector_store: str = "chromadb"  # chromadb or sqlite-vss
    chroma_path: str = "./data/chromadb"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


@dataclass
class ProcessingConfig:
    """Processing configuration"""
    max_workers: int = 4
    batch_size: int = 50
    use_hardlinks: bool = True
    enable_deduplication: bool = True
    enable_optimization: bool = False
    enable_par2: bool = False
    
    # Burst culling
    burst_time_threshold: float = 1.0  # seconds
    burst_keep_count: int = 1
    
    # Privacy vault
    enable_vault: bool = True
    vault_encrypted: bool = False


@dataclass
class WatchdogConfig:
    """File watchdog configuration"""
    enabled: bool = True
    debounce_time: float = 2.0  # seconds
    recursive: bool = True
    ignore_patterns: list = field(default_factory=lambda: [
        ".*",  # Hidden files
        "*.tmp", "*.temp", "*.bak", "*.backup",
        "Thumbs.db", "desktop.ini", ".DS_Store",
        "*.lnk", "*.url"  # Windows shortcuts
    ])
    ignore_directories: list = field(default_factory=lambda: [
        "@eaDir", ".thumbnails", ".trash", ".stfolder"
    ])


@dataclass
class WebConfig:
    """Web interface configuration"""
    # FastAPI
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    cors_origins: list = field(default_factory=lambda: [
        "http://localhost:8000",
        "http://localhost:8501"
    ])
    
    # Streamlit
    streamlit_port: int = 8501
    streamlit_theme: str = "dark"
    
    # Authentication
    enable_auth: bool = False
    jwt_secret: str = ""
    session_timeout: int = 3600  # seconds

@dataclass
class SystemConfig:
    """System-level paths and runtime settings"""
    data_dir: Path = Path.home() / ".mnemosyne" / "data"
    temp_dir: Path = Path.home() / ".mnemosyne" / "temp"
    cache_dir: Path = Path.home() / ".mnemosyne" / "cache"
    models_dir: Path = Path.home() / ".mnemosyne" / "models"

    def __post_init__(self):
        for field_name in ["data_dir", "temp_dir", "cache_dir", "models_dir"]:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

@dataclass
class Config:
    """Main configuration class"""
    # System
    system: SystemConfig = field(default_factory=SystemConfig)
    system_name: str = "Mnemosyne"
    version: str = "1.0.0"
    platform: str = "auto"  # auto, windows, linux, macos
    
    # Paths
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Modules
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)
    web: WebConfig = field(default_factory=WebConfig)
    
    # Advanced
    log_level: str = "INFO"
    log_file: str = "./logs/mnemosyne.log"
    backup_enabled: bool = True
    backup_interval: int = 86400  # seconds (24 hours)
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    
    def __post_init__(self):
        # Auto-detect platform if not set
        if self.platform == "auto":
            import sys
            if sys.platform == "win32":
                self.platform = "windows"
            elif sys.platform == "darwin":
                self.platform = "macos"
            else:
                self.platform = "linux"
        
        # Adjust paths based on platform
        self._adjust_paths_for_platform()
    
    def _adjust_paths_for_platform(self):
        """Adjust paths based on the detected platform"""
        import sys
        from pathlib import Path
        
        if self.platform == "windows":
            import os
            appdata = Path(os.environ.get('LOCALAPPDATA', Path.home()))
            self.paths.vault = appdata / "Mnemosyne" / "vault"
            self.paths.cache = appdata / "Mnemosyne" / "cache"
            self.paths.temp = Path("C:/Temp/Mnemosyne")
            self.paths.models = appdata / "Mnemosyne" / "models"
        elif self.platform == "macos":
            self.paths.vault = Path.home() / "Library" / "Application Support" / "Mnemosyne" / "vault"
            self.paths.cache = Path.home() / "Library" / "Caches" / "Mnemosyne"
            self.paths.temp = Path("/tmp/mnemosyne")
            self.paths.models = Path.home() / ".mnemosyne" / "models"
        else:  # linux
            self.paths.vault = Path.home() / ".local" / "share" / "mnemosyne" / "vault"
            self.paths.cache = Path.home() / ".cache" / "mnemosyne"
            self.paths.temp = Path("/tmp/mnemosyne")
            self.paths.models = Path.home() / ".mnemosyne" / "models"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        def serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in asdict(obj).items()}
            else:
                return obj
        
        return serialize(asdict(self))
    
    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def to_yaml(self) -> str:
        """Convert config to YAML string"""
        import yaml
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(self, path: Union[str, Path]):
        """Save config to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if path.suffix.lower() in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:  # default to JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {path}")
    
    def update_from_dict(self, data: Dict[str, Any]):
        """Update config from dictionary"""
        # This is a simplified update - in production you'd want more robust merging
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif hasattr(self.paths, key):
                setattr(self.paths, key, value)
            elif hasattr(self.ollama, key):
                setattr(self.ollama, key, value)
            elif hasattr(self.ai, key):
                setattr(self.ai, key, value)
            elif hasattr(self.database, key):
                setattr(self.database, key, value)
            elif hasattr(self.processing, key):
                setattr(self.processing, key, value)
            elif hasattr(self.watchdog, key):
                setattr(self.watchdog, key, value)
            elif hasattr(self.web, key):
                setattr(self.web, key, value)


def load_config(path: Union[str, Path] = None) -> Config:
    """
    Load configuration from file or create default
    """
    config_paths = []
    
    # Check multiple possible locations
    if path:
        config_paths.append(Path(path))
    
    # Platform-specific default locations
    import sys
    if sys.platform == "win32":
        import os
        appdata = Path(os.environ.get('LOCALAPPDATA', Path.home()))
        config_paths.extend([
            Path("config.yaml"),
            Path("config.json"),
            appdata / "Mnemosyne" / "config.yaml",
            appdata / "Mnemosyne" / "config.json",
        ])
    elif sys.platform == "darwin":
        config_paths.extend([
            Path("config.yaml"),
            Path("config.json"),
            Path.home() / "Library" / "Application Support" / "Mnemosyne" / "config.yaml",
            Path.home() / "Library" / "Application Support" / "Mnemosyne" / "config.json",
        ])
    else:  # linux
        config_paths.extend([
            Path("config.yaml"),
            Path("config.json"),
            Path.home() / ".config" / "mnemosyne" / "config.yaml",
            Path.home() / ".config" / "mnemosyne" / "config.json",
            Path("/etc/mnemosyne/config.yaml"),
        ])
    
    # Try to load from existing config file
    for config_path in config_paths:
        if config_path.exists():
            try:
                logger.info(f"Loading configuration from {config_path}")
                
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                else:  # JSON
                    with open(config_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                # Create config and update with loaded data
                config = Config()
                config.update_from_dict(data)
                
                # Adjust paths for platform
                config._adjust_paths_for_platform()
                
                logger.info("Configuration loaded successfully")
                return config
                
            except Exception as e:
                logger.error(f"Error loading configuration from {config_path}: {e}")
    
    # No config found, create default
    logger.info("No configuration file found, creating default configuration")
    config = Config()
    
    # Save default config to first location
    if config_paths:
        config.save(config_paths[0])
    
    return config


def save_config(config: Config, path: Union[str, Path] = None):
    """Save configuration to file"""
    if path is None:
        # Use platform-specific default location
        import sys
        if sys.platform == "win32":
            import os
            appdata = Path(os.environ.get('LOCALAPPDATA', Path.home()))
            path = appdata / "Mnemosyne" / "config.yaml"
        elif sys.platform == "darwin":
            path = Path.home() / "Library" / "Application Support" / "Mnemosyne" / "config.yaml"
        else:  # linux
            path = Path.home() / ".config" / "mnemosyne" / "config.yaml"
    
    config.save(path)


def get_default_config_path() -> Path:
    """Get default configuration path based on platform"""
    import sys
    if sys.platform == "win32":
        import os
        appdata = Path(os.environ.get('LOCALAPPDATA', Path.home()))
        return appdata / "Mnemosyne" / "config.yaml"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Mnemosyne" / "config.yaml"
    else:  # linux
        return Path.home() / ".config" / "mnemosyne" / "config.yaml"


# Global config instance
_config_instance = None

def get_config() -> Config:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def reload_config(path: Union[str, Path] = None) -> Config:
    """Reload configuration from file"""
    global _config_instance
    _config_instance = load_config(path)
    return _config_instance