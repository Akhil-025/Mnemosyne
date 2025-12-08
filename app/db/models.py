#app/db/models.py

"""
SQLAlchemy models for Mnemosyne database
"""
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, LargeBinary, BigInteger
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.dialects.sqlite import BLOB
from datetime import datetime
from typing import Optional, List
import json
from pathlib import Path

Base = declarative_base()


class File(Base):
    """Media file metadata"""
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String(1024), unique=True, nullable=False, index=True)
    file_hash = Column(String(64), nullable=False, index=True)  # SHA256
    perceptual_hash = Column(String(64), index=True)  # pHash for visual similarity
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(10), nullable=False)  # image, video, audio
    mime_type = Column(String(100))
    
    # Processing status
    ingested_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime)
    processed = Column(Boolean, default=False)
    error = Column(Text)
    
    # Relationships
    embeddings = relationship("Embedding", back_populates="file", cascade="all, delete-orphan")
    faces = relationship("FaceDetection", back_populates="file", cascade="all, delete-orphan")
    events = relationship("Event", secondary="event_files", back_populates="files")

    # Optional: basic analysis fields mirrored on File
    caption = Column(Text, nullable=True)
    ocr_text = Column(Text, nullable=True)
    objects = Column(JSON, nullable=True)
    tags = Column(JSON, nullable=True)
    mood = Column(String, nullable=True)
    is_sensitive = Column(Boolean, default=False)

    blur_score = Column(Float, nullable=True)
    color_palette = Column(JSON, nullable=True)

    # new flag: enqueue for analysis if True (currently unused in your worker)
    needs_analysis = Column(Boolean, default=True, index=True)
    
    def __repr__(self):
        return f"<File(id={self.id}, path={self.file_path}, type={self.file_type})>"


class Event(Base):
    """Temporal/spatial clusters of photos (events)"""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    description = Column(Text)
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime, index=True)
    location_name = Column(String(255))
    latitude = Column(Float)
    longitude = Column(Float)
    radius_km = Column(Float)  # Cluster radius in km
    file_count = Column(Integer, default=0)
    cover_file_id = Column(Integer, ForeignKey('files.id'))
    
    # Relationships
    files = relationship("File", secondary="event_files", back_populates="events")
    cover_file = relationship("File", foreign_keys=[cover_file_id])
    
    def __repr__(self):
        return f"<Event(id={self.id}, name={self.name}, files={self.file_count})>"


class EventFile(Base):
    """Association table for events and files"""
    __tablename__ = 'event_files'
    
    event_id = Column(Integer, ForeignKey('events.id'), primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), primary_key=True)
    sequence = Column(Integer)  # Order within event


class Embedding(Base):
    """Vector embeddings for semantic search"""
    __tablename__ = 'embeddings'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False, index=True)
    embedding_type = Column(String(50), nullable=False)  # 'caption', 'visual', 'clip'
    embedding_vector = Column(LargeBinary, nullable=False)  # Serialized numpy array
    dimensions = Column(Integer, nullable=False)
    model_name = Column(String(100))
    generated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    file = relationship("File", back_populates="embeddings")
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, file_id={self.file_id}, type={self.embedding_type})>"


class FaceDetection(Base):
    """Detected faces in media files"""
    __tablename__ = 'face_detections'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False, index=True)
    person_id = Column(Integer, ForeignKey('persons.id'), index=True)
    
    # Bounding box
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Face embedding
    embedding = Column(LargeBinary, nullable=False)  # 512-dim InsightFace embedding
    embedding_model = Column(String(50))
    
    # Attributes
    confidence = Column(Float)
    gender = Column(String(1))  # 'M', 'F', None
    age = Column(Integer)
    eye_open_prob = Column(Float)  # Probability eyes are open
    
    # Clustering
    cluster_id = Column(Integer, index=True)
    
    # Relationships
    file = relationship("File", back_populates="faces")
    person = relationship("Person", back_populates="detections")
    
    def __repr__(self):
        return f"<FaceDetection(id={self.id}, file_id={self.file_id}, person_id={self.person_id})>"


class Person(Base):
    """Identified persons"""
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), index=True)
    custom_name = Column(String(100))
    
    # Representative embedding
    representative_embedding = Column(LargeBinary)
    
    # Statistics
    first_seen = Column(DateTime)
    last_seen = Column(DateTime)
    photo_count = Column(Integer, default=0)
    face_count = Column(Integer, default=0)
    
    # Metadata
    extra_metadata = Column(JSON)  # Custom metadata (birthday, relationships, etc.)
    
    # Relationships
    detections = relationship("FaceDetection", back_populates="person")
    
    def __repr__(self):
        return f"<Person(id={self.id}, name={self.name or 'Unknown'}, photos={self.photo_count})>"


class AnalysisResult(Base):
    """AI analysis results for media files"""
    __tablename__ = 'analysis_results'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False, unique=True, index=True)
    
    # Ollama analysis
    caption = Column(Text)
    tags = Column(JSON)  # List of tags
    objects = Column(JSON)  # List of detected objects
    mood = Column(String(50))
    contains_text = Column(Boolean)
    is_sensitive = Column(Boolean)
    
    # Technical analysis
    aesthetic_score = Column(Float)
    sharpness = Column(Float)
    color_palette = Column(JSON)  # List of dominant colors
    
    # Video specific
    video_duration = Column(Float)
    scene_changes = Column(JSON)  # List of scene change timestamps
    
    # GPS data
    latitude = Column(Float)
    longitude = Column(Float)
    altitude = Column(Float)
    location_name = Column(String(255))
    
    # Timestamps
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analysis_version = Column(String(50))
    
    # Relationships
    file = relationship("File")
    
    def __repr__(self):
        return f"<AnalysisResult(id={self.id}, file_id={self.file_id})>"


class ProcessingQueue(Base):
    """Background processing queue"""
    __tablename__ = 'processing_queue'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), index=True)
    task_type = Column(String(50), nullable=False)  # 'analyze', 'embed', 'cluster_faces'
    priority = Column(Integer, default=0)  # Higher = more important
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error = Column(Text)
    
    # Relationships
    file = relationship("File")
    
    def __repr__(self):
        return f"<ProcessingQueue(id={self.id}, file_id={self.file_id}, task={self.task_type})>"
    
class IngestionLog(Base):
    __tablename__ = "ingestion_logs"

    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    finished_at = Column(DateTime)
    ingested = Column(Integer, default=0)
    duplicates = Column(Integer, default=0)
    errors = Column(Integer, default=0)
    total_size = Column(BigInteger, default=0)
    details = Column(JSON, nullable=True)  # batch/file breakdown


# Create tables
def create_tables(engine):
    """Create all database tables"""
    Base.metadata.create_all(engine)