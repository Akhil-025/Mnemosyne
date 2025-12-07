# app/db/vector_store.py

"""
Vector storage and similarity search using sqlite-vss
"""
import sqlite3
import numpy as np
from typing import List, Optional, Tuple, Any
from pathlib import Path
import json
import pickle
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with similarity score"""
    file_id: int
    file_path: str
    similarity: float
    metadata: dict


class VectorStore:
    """Vector similarity search using sqlite-vss"""
    
    def __init__(self, db_path: Path, dimensions: int = 768):
        self.db_path = db_path
        self.dimensions = dimensions
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize vector database with sqlite-vss extension"""
        # Create directory if it doesn't exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.enable_load_extension(True)
        
        try:
            # Load sqlite-vss extension
            # Note: This requires the extension to be compiled/available
            self.conn.load_extension("vector0")
            self.conn.load_extension("vss0")
            logger.info("sqlite-vss extensions loaded successfully")
        except sqlite3.OperationalError as e:
            logger.error(f"Failed to load sqlite-vss extensions: {e}")
            logger.info("Falling back to manual vector operations")
            self.use_vss = False
        else:
            self.use_vss = True
        
        # Create tables
        self._create_tables()
    
    def _create_tables(self):
        """Create vector tables"""
        cursor = self.conn.cursor()
        
        if self.use_vss:
            # Create vss0 virtual table for vector search
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_embeddings USING vss0(
                    embedding({self.dimensions})
                )
            """)
        
        # Create regular table for metadata
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL,
                embedding_type TEXT NOT NULL,
                embedding_blob BLOB NOT NULL,
                dimensions INTEGER NOT NULL,
                model_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_id, embedding_type)
            )
        """)
        
        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_file_id 
            ON embeddings(file_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_embeddings_type 
            ON embeddings(embedding_type)
        """)
        
        self.conn.commit()
    
    def add_embedding(self, file_id: int, embedding_type: str, 
                     embedding: np.ndarray, model_name: str = None) -> bool:
        """Add embedding to vector store"""
        try:
            # Serialize embedding
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            embedding_blob = embedding.tobytes()
            dimensions = len(embedding)
            
            cursor = self.conn.cursor()
            
            # Insert into embeddings table
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings 
                (file_id, embedding_type, embedding_blob, dimensions, model_name)
                VALUES (?, ?, ?, ?, ?)
            """, (file_id, embedding_type, embedding_blob, dimensions, model_name))
            
            if self.use_vss:
                # Insert into vss table
                cursor.execute("""
                    INSERT OR REPLACE INTO vss_embeddings(rowid, embedding)
                    VALUES (?, ?)
                """, (cursor.lastrowid, embedding_blob))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, embedding_type: str,
                      limit: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar embeddings"""
        try:
            if isinstance(query_embedding, list):
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            query_blob = query_embedding.tobytes()
            
            if self.use_vss:
                # Use vss0 for similarity search
                cursor = self.conn.cursor()
                cursor.execute("""
                    SELECT 
                        e.file_id,
                        vss_distance(vss_embeddings.embedding, ?) as distance
                    FROM vss_embeddings
                    JOIN embeddings e ON vss_embeddings.rowid = e.id
                    WHERE e.embedding_type = ?
                    ORDER BY distance
                    LIMIT ?
                """, (query_blob, embedding_type, limit))
                
                results = cursor.fetchall()
                
                # Convert distance to similarity (assuming cosine distance)
                # vss_distance returns squared L2 distance
                similarities = [(file_id, 1.0 / (1.0 + distance)) 
                              for file_id, distance in results]
            else:
                # Manual cosine similarity calculation
                cursor = self.conn.cursor()
                cursor.execute("""
                    SELECT file_id, embedding_blob 
                    FROM embeddings 
                    WHERE embedding_type = ?
                """, (embedding_type,))
                
                similarities = []
                query_norm = np.linalg.norm(query_embedding)
                
                for file_id, embedding_blob in cursor.fetchall():
                    emb = np.frombuffer(embedding_blob, dtype=np.float32)
                    emb_norm = np.linalg.norm(emb)
                    
                    if query_norm > 0 and emb_norm > 0:
                        similarity = np.dot(query_embedding, emb) / (query_norm * emb_norm)
                        similarities.append((file_id, float(similarity)))
                
                # Sort by similarity
                similarities.sort(key=lambda x: x[1], reverse=True)
                similarities = similarities[:limit]
            
            # Filter by threshold and fetch file paths
            results = []
            cursor = self.conn.cursor()
            
            for file_id, similarity in similarities:
                if similarity >= threshold:
                    cursor.execute("""
                        SELECT file_path FROM files WHERE id = ?
                    """, (file_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        results.append(SearchResult(
                            file_id=file_id,
                            file_path=row[0],
                            similarity=similarity,
                            metadata={}
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    def get_embedding(self, file_id: int, embedding_type: str) -> Optional[np.ndarray]:
        """Retrieve embedding for a file"""
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT embedding_blob, dimensions 
                FROM embeddings 
                WHERE file_id = ? AND embedding_type = ?
            """, (file_id, embedding_type))
            
            row = cursor.fetchone()
            if row:
                embedding_blob, dimensions = row
                return np.frombuffer(embedding_blob, dtype=np.float32)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
    
    def delete_embeddings(self, file_id: int) -> bool:
        """Delete all embeddings for a file"""
        try:
            cursor = self.conn.cursor()
            
            # Get embedding IDs for vss table
            cursor.execute("""
                SELECT id FROM embeddings WHERE file_id = ?
            """, (file_id,))
            
            embedding_ids = [row[0] for row in cursor.fetchall()]
            
            # Delete from vss table
            if self.use_vss and embedding_ids:
                placeholders = ','.join('?' * len(embedding_ids))
                cursor.execute(f"""
                    DELETE FROM vss_embeddings 
                    WHERE rowid IN ({placeholders})
                """, embedding_ids)
            
            # Delete from embeddings table
            cursor.execute("DELETE FROM embeddings WHERE file_id = ?", (file_id,))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get vector store statistics"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT file_id) FROM embeddings")
        unique_files = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT embedding_type) FROM embeddings")
        embedding_types = cursor.fetchone()[0]
        
        return {
            'total_embeddings': total_embeddings,
            'unique_files': unique_files,
            'embedding_types': embedding_types,
            'using_vss': self.use_vss
        }
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()


class HybridVectorStore:
    """Hybrid vector store using ChromaDB as fallback"""
    
    def __init__(self, storage_path: Path, dimensions: int = 768):
        self.storage_path = storage_path
        self.dimensions = dimensions
        
        # Try sqlite-vss first
        self.primary_store = VectorStore(storage_path / "vectors.db", dimensions)
        
        # If sqlite-vss failed, use ChromaDB
        if not self.primary_store.use_vss:
            logger.info("Falling back to ChromaDB")
            self._init_chromadb()
        else:
            self.chroma_client = None
    
    def _init_chromadb(self):
        """Initialize ChromaDB"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.storage_path / "chromadb"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="media_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            
        except ImportError:
            logger.error("ChromaDB not installed")
            self.chroma_client = None
            raise
    
    def add_embedding(self, file_id: int, embedding_type: str, 
                     embedding: np.ndarray, model_name: str = None) -> bool:
        """Add embedding to store"""
        if self.chroma_client:
            return self._add_to_chromadb(file_id, embedding_type, embedding, model_name)
        else:
            return self.primary_store.add_embedding(file_id, embedding_type, embedding, model_name)
    
    def _add_to_chromadb(self, file_id: int, embedding_type: str, 
                        embedding: np.ndarray, model_name: str) -> bool:
        """Add embedding to ChromaDB"""
        try:
            # Convert to list
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            # Create unique ID
            embedding_id = f"{file_id}_{embedding_type}"
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding],
                metadatas=[{
                    "file_id": file_id,
                    "embedding_type": embedding_type,
                    "model_name": model_name or "unknown"
                }],
                ids=[embedding_id]
            )
            return True
            
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {e}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, embedding_type: str,
                      limit: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Search for similar embeddings"""
        if self.chroma_client:
            return self._search_chromadb(query_embedding, embedding_type, limit, threshold)
        else:
            return self.primary_store.search_similar(query_embedding, embedding_type, limit, threshold)
    
    def _search_chromadb(self, query_embedding: np.ndarray, embedding_type: str,
                        limit: int, threshold: float) -> List[SearchResult]:
        """Search in ChromaDB"""
        try:
            # Convert to list
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where={"embedding_type": embedding_type},
                include=["metadatas", "distances"]
            )
            
            # Convert results
            search_results = []
            if results['ids'] and results['distances']:
                for ids, distances, metadatas in zip(results['ids'], results['distances'], results['metadatas']):
                    for id, distance, metadata in zip(ids[0], distances[0], metadatas[0]):
                        # Convert distance to similarity
                        similarity = 1.0 - distance
                        
                        if similarity >= threshold:
                            search_results.append(SearchResult(
                                file_id=metadata.get("file_id", 0),
                                file_path="",  # Would need to fetch from main DB
                                similarity=similarity,
                                metadata=metadata
                            ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return []