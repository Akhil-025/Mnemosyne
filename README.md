Project Architecture


Mnemosyne/
├── app/
│   ├── core/              # Core system components
│   │   ├── __init__.py
│   │   ├── ollama_client.py      # Ollama API wrapper
│   │   ├── face_analysis.py      # InsightFace face analysis
│   │   ├── video_processor.py    # Video frame extraction
│   │   └── intelligence_engine.py # Main AI orchestration
│   │
│   ├── db/                # Data layer
│   │   ├── __init__.py
│   │   ├── models.py      # SQLAlchemy models
│   │   ├── vector_store.py # Vector search with sqlite-vss
│   │   ├── schema.sql     # SQL schema
│   │   └── session.py     # Database session management
│   │
│   ├── processing/        # File processing pipelines
│   │   ├── __init__.py
│   │   ├── ingestion.py   # File ingestion & deduplication
│   │   ├── analysis.py    # AI analysis pipeline
│   │   ├── burst_culling.py # Smart burst selection
│   │   ├── metadata_writer.py # EXIF/IPTC write-back
│   │   ├── privacy_vault.py # Sensitive content detection
│   │   └── workflow.py #ties everything together with a task scheduler that can prioritize and manage different processing tasks efficiently
│   │
│   ├── web/              # Interaction layer
│   │   ├── __init__.py
│   │   ├── api.py        # FastAPI endpoints
│   │   ├── rag.py        # RAG system for chat
│   │   ├── streamlit_app.py # Streamlit dashboard
│   │   └── static/       # Frontend assets
│   │
│   ├── tasks/            # Background processing
│   │   ├── __init__.py
│   │   ├── queue.py      # Priority task queue
│   │   ├── worker.py     # Celery workers
│   │   └── scheduler.py  # Scheduled tasks
│   │
│   ├── utils/            # Utilities
│   │   ├── __init__.py
│   │   ├── config.py           # Complete configuration management with platform-specific defaults
│   │   ├── logger.py           # Structured logging with JSON, color, and performance logging
│   │   ├── file_utils.py       # File operations, type detection, hashing, and metadata
│   │   ├── geocoding.py        # Reverse geocoding with caching and offline support
│   │   ├── windows_compat.py   # Windows-specific utilities and compatibility layer
│   │   └── image_utils.py      # Image processing, analysis, and feature extraction
│   │
│   └── watchdog/         # File monitoring
│       ├── __init__.py
│       ├── events.py       # Shared event models (EventType, WatchdogEvent) used across all watchdog modules
│       ├── monitor.py      # Main file system monitor with debouncing and event processing
│       ├── debounce.py     # Advanced event debouncing with batch processing
│       ├── patterns.py     # Pattern matching and filtering for ignoring files/directories
│       ├── watcher.py      # Directory watchers with recursive monitoring and adaptive polling
│       └── handlers.py     # Event handlers with different strategies (immediate, batched)
│
├── data/                 # Application data
│   ├── vault/           # Encrypted privacy vault
│   ├── thumbnails/      # Generated thumbnails
│   └── cache/           # Temporary cache
│
├── tests/               # Test suite
├── docker-compose.yml   # Container orchestration
├── Dockerfile           # Container definition
├── requirements.txt     # Python dependencies
├── start_mnemosyne.bat
├── mnemosyne_windows.log
├── config.yaml          # Application configuration
└── README.md           # Documentation

