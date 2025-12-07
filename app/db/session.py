#app/db/session.py

"""
Database session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import logging
from pathlib import Path
from typing import Generator
from .models import Base, create_tables

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manage database connections and sessions"""
    
    def __init__(self, db_url: str = "sqlite:///./data/mnemosyne.db", echo: bool = False):
        self.db_url = db_url
        self.engine = create_engine(
            db_url,
            echo=echo,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        # Create all tables
        create_tables(self.engine)
        logger.info(f"Database initialized at {self.db_url}")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session context manager"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_db(self):
        """FastAPI dependency for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close()


# Global database manager instance
_db_manager = None

def get_db_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    if _db_manager is None:
        # Create data directory if it doesn't exist
        Path("./data").mkdir(exist_ok=True)
        _db_manager = DatabaseManager()
    return _db_manager


def init_database(db_url: str = None, echo: bool = False):
    """Initialize database with custom URL"""
    global _db_manager
    if db_url:
        _db_manager = DatabaseManager(db_url, echo)
    else:
        _db_manager = DatabaseManager(echo=echo)
    return _db_manager