--app/db/schema.sql

session.py -- Mnemosyne Database Schema
-- SQLite with sqlite-vss extension for vector search

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Files table
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL UNIQUE,
    file_hash TEXT NOT NULL,
    perceptual_hash TEXT,
    file_size INTEGER NOT NULL,
    file_type TEXT NOT NULL CHECK(file_type IN ('image', 'video', 'audio')),
    mime_type TEXT,
    
    -- Processing status
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analyzed_at TIMESTAMP,
    processed BOOLEAN DEFAULT 0,
    error TEXT,
    
    -- Indexes
    INDEX idx_files_hash (file_hash),
    INDEX idx_files_type (file_type),
    INDEX idx_files_processed (processed)
);

-- Events table (temporal/spatial clusters)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    location_name TEXT,
    latitude REAL,
    longitude REAL,
    radius_km REAL,
    file_count INTEGER DEFAULT 0,
    cover_file_id INTEGER,
    
    -- Foreign key
    FOREIGN KEY (cover_file_id) REFERENCES files(id) ON DELETE SET NULL,
    
    -- Indexes
    INDEX idx_events_time (start_time, end_time),
    INDEX idx_events_location (latitude, longitude)
);

-- Event-Files association table
CREATE TABLE IF NOT EXISTS event_files (
    event_id INTEGER NOT NULL,
    file_id INTEGER NOT NULL,
    sequence INTEGER,
    
    PRIMARY KEY (event_id, file_id),
    FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    
    INDEX idx_event_files_file (file_id)
);

-- Embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    embedding_type TEXT NOT NULL,
    embedding_blob BLOB NOT NULL,
    dimensions INTEGER NOT NULL,
    model_name TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(file_id, embedding_type),
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    
    INDEX idx_embeddings_file (file_id),
    INDEX idx_embeddings_type (embedding_type)
);

-- Persons table (identified people)
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    custom_name TEXT,
    representative_embedding BLOB,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP,
    photo_count INTEGER DEFAULT 0,
    face_count INTEGER DEFAULT 0,
    metadata TEXT,  -- JSON metadata
    
    INDEX idx_persons_name (name)
);

-- Face detections table
CREATE TABLE IF NOT EXISTS face_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL,
    person_id INTEGER,
    
    -- Bounding box
    bbox_x1 INTEGER,
    bbox_y1 INTEGER,
    bbox_x2 INTEGER,
    bbox_y2 INTEGER,
    
    -- Face embedding
    embedding BLOB NOT NULL,
    embedding_model TEXT,
    
    -- Attributes
    confidence REAL,
    gender TEXT CHECK(gender IN ('M', 'F')),
    age INTEGER,
    eye_open_prob REAL,
    
    -- Clustering
    cluster_id INTEGER,
    
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL,
    
    INDEX idx_faces_file (file_id),
    INDEX idx_faces_person (person_id),
    INDEX idx_faces_cluster (cluster_id)
);

-- Analysis results table
CREATE TABLE IF NOT EXISTS analysis_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL UNIQUE,
    
    -- Ollama analysis
    caption TEXT,
    tags TEXT,  -- JSON array
    objects TEXT,  -- JSON array
    mood TEXT,
    contains_text BOOLEAN DEFAULT 0,
    is_sensitive BOOLEAN DEFAULT 0,
    
    -- Technical analysis
    aesthetic_score REAL,
    sharpness REAL,
    color_palette TEXT,  -- JSON array
    
    -- Video specific
    video_duration REAL,
    scene_changes TEXT,  -- JSON array
    
    -- GPS data
    latitude REAL,
    longitude REAL,
    altitude REAL,
    location_name TEXT,
    
    -- Timestamps
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_version TEXT,
    
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    
    INDEX idx_analysis_file (file_id)
);

-- Processing queue table
CREATE TABLE IF NOT EXISTS processing_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER,
    task_type TEXT NOT NULL,
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error TEXT,
    
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE,
    
    INDEX idx_queue_status (status),
    INDEX idx_queue_priority (priority DESC),
    INDEX idx_queue_file (file_id)
);

-- System settings table
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create vector search virtual table using sqlite-vss
-- Note: This requires the sqlite-vss extension to be loaded
-- CREATE VIRTUAL TABLE IF NOT EXISTS vss_embeddings USING vss0(
--     embedding(768)
-- );

-- Insert default settings
INSERT OR IGNORE INTO settings (key, value) VALUES 
    ('database_version', '1.0.0'),
    ('ollama_url', 'http://localhost:11434'),
    ('ollama_image_model', 'llava:13b'),
    ('ollama_embedding_model', 'nomic-embed-text:latest'),
    ('face_model', 'buffalo_l'),
    ('vector_dimensions', '768');