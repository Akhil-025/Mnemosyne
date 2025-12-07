# Mnemosyne
*She does not command. She remembers.*

---

## Project Architecture

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
│   │   └── workflow.py    # Task scheduler and orchestration
│   │
│   ├── web/               # Interaction layer
│   │   ├── __init__.py
│   │   ├── api.py         # FastAPI endpoints
│   │   ├── rag.py         # RAG system for chat
│   │   ├── streamlit_app.py # Streamlit dashboard
│   │   └── static/        # Frontend assets
│   │
│   ├── tasks/             # Background processing
│   │   ├── __init__.py
│   │   ├── queue.py       # Priority task queue
│   │   ├── worker.py      # Celery workers
│   │   └── scheduler.py   # Scheduled tasks
│   │
│   ├── utils/             # Utilities
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── logger.py         # Structured logging
│   │   ├── file_utils.py     # File operations & hashing
│   │   ├── geocoding.py      # Reverse geocoding
│   │   ├── windows_compat.py # Windows compatibility
│   │   └── image_utils.py    # Image processing
│   │
│   └── watchdog/          # File monitoring
│       ├── __init__.py
│       ├── events.py
│       ├── monitor.py
│       ├── debounce.py
│       ├── patterns.py
│       ├── watcher.py
│       └── handlers.py
│
├── data/                  # Application data (ignored via .gitignore)
├── tests/                 # Test suite
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── start_mnemosyne.bat
├── config.yaml
