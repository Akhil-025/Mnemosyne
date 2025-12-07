# app/processing/ingestion.py

"""
File ingestion and deduplication module
"""
import hashlib
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import shutil
import os

from sqlalchemy.orm import Session

from app.db.models import File, ProcessingQueue
from app.utils.file_utils import get_file_type, is_image_file, is_video_file
from app.utils.windows_compat import is_windows, create_windows_junction

logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """File information for ingestion"""
    path: Path
    size: int
    mtime: datetime
    hash: str
    perceptual_hash: Optional[str] = None
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    error: Optional[str] = None


class DuplicateDetector:
    """Detect duplicate files using multiple strategies"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        
    async def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(65536), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {file_path}: {e}")
            raise
    
    async def compute_perceptual_hash(self, file_path: Path) -> Optional[str]:
        """Compute perceptual hash for visual similarity (images only)"""
        if not is_image_file(file_path):
            return None
        
        try:
            import imagehash
            from PIL import Image
            
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode not in ['RGB', 'L']:
                    img = img.convert('RGB')
                
                # Compute average hash
                ahash = str(imagehash.average_hash(img))
                
                # Compute perceptual hash
                phash = str(imagehash.phash(img))
                
                # Combine hashes
                return f"{ahash}:{phash}"
                
        except ImportError:
            logger.warning("imagehash not installed, perceptual hashing disabled")
            return None
        except Exception as e:
            logger.error(f"Error computing perceptual hash for {file_path}: {e}")
            return None
    
    async def find_duplicates(self, file_path: Path, file_hash: str, 
                            perceptual_hash: Optional[str] = None) -> List[Tuple[int, str, float]]:
        """Find duplicate files in database."""
        
        # 0. If EXACT same path already exists, treat as already ingested
        existing_path = self.db.query(File).filter(
            File.file_path == str(file_path)
        ).first()

        if existing_path:
            # Return it as a 100% duplicate so ingestion can skip it
            return [(existing_path.id, existing_path.file_path, 1.0)]
        
        duplicates = []

        # 1. Exact hash duplicates
        existing = self.db.query(File).filter(File.file_hash == file_hash).all()
        for file in existing:
            if Path(file.file_path).resolve() != file_path.resolve():
                duplicates.append((file.id, file.file_path, 1.0))

        # 2. Perceptual duplicates (if available)
        if perceptual_hash and not duplicates:
            existing_visual = self.db.query(File).filter(
                File.perceptual_hash.isnot(None)
            ).all()

            for file in existing_visual:
                if file.perceptual_hash and self._compare_perceptual_hashes(
                    perceptual_hash, file.perceptual_hash, threshold=0.9
                ):
                    duplicates.append((file.id, file.file_path, 0.9))

        return duplicates

    
    def _compare_perceptual_hashes(self, hash1: str, hash2: str, threshold: float = 0.9) -> bool:
        """Compare perceptual hashes for similarity"""
        try:
            # Simple comparison - in production use proper distance metrics
            if not hash1 or not hash2:
                return False
            
            # Extract hash components
            hash1_parts = hash1.split(':')
            hash2_parts = hash2.split(':')
            
            if len(hash1_parts) != len(hash2_parts):
                return False
            
            # Compare each hash component
            matches = 0
            for h1, h2 in zip(hash1_parts, hash2_parts):
                # Hamming distance for image hashes
                if len(h1) == len(h2):
                    distance = sum(c1 != c2 for c1, c2 in zip(h1, h2))
                    similarity = 1.0 - (distance / len(h1))
                    if similarity >= threshold:
                        matches += 1
            
            return matches >= 1  # At least one hash component matches
            
        except Exception as e:
            logger.error(f"Error comparing perceptual hashes: {e}")
            return False
    
    async def create_hardlink(self, source: Path, target: Path) -> bool:
        """Create hardlink or copy file depending on platform and filesystem"""
        try:
            if is_windows():
                # On Windows, check if same drive
                if source.drive == target.drive:
                    try:
                        os.link(source, target)
                        return True
                    except OSError:
                        # Fallback to junction for directories, copy for files
                        if source.is_dir():
                            return create_windows_junction(source, target)
                        else:
                            shutil.copy2(source, target)
                            return True
                else:
                    # Different drives, copy
                    shutil.copy2(source, target)
                    return True
            else:
                # Unix-like: try hardlink
                try:
                    os.link(source, target)
                    return True
                except OSError:
                    # Fallback to copy
                    shutil.copy2(source, target)
                    return True
                    
        except Exception as e:
            logger.error(f"Error creating hardlink/copy from {source} to {target}: {e}")
            return False


class FileIngestor:
    """Handle file ingestion and organization"""
    
    def __init__(self, config: Dict, db_manager):
        self.config = config
        self.db_manager = db_manager
        self.duplicate_detector = None
        self.stats = {
            'ingested': 0,
            'duplicates_skipped': 0,
            'errors': 0,
            'total_size': 0
        }
    
    async def process_directory(self, directory: Path, recursive: bool = True) -> Dict:
        """Process all files in directory"""
        logger.info(f"Processing directory: {directory}")
        
        with self.db_manager.get_session() as session:
            self.duplicate_detector = DuplicateDetector(session)
            
            # Find all media files
            file_paths = self._find_media_files(directory, recursive)
            logger.info(f"Found {len(file_paths)} files to process")
            
            # Process in batches
            batch_size = self.config.processing.batch_size
            for i in range(0, len(file_paths), batch_size):
                batch = file_paths[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
                
                await self._process_batch(batch, session)
            
            session.commit()
        
        return self.stats.copy()
    
    def _find_media_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all media files in directory"""
        media_extensions = {
            # Images
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif',
            '.webp', '.heic', '.heif', '.raw', '.nef', '.cr2', '.arw',
            '.dng', '.orf', '.sr2',
            # Videos
            '.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv',
            '.m4v', '.mpg', '.mpeg', '.3gp', '.mts', '.m2ts',
            # Audio
            '.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'
        }
        
        files = []
        
        if recursive:
            for ext in media_extensions:
                files.extend(directory.rglob(f"*{ext}"))
                files.extend(directory.rglob(f"*{ext.upper()}"))
        else:
            for ext in media_extensions:
                files.extend(directory.glob(f"*{ext}"))
                files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Filter out system/hidden files
        filtered_files = []
        for file_path in files:
            # Skip hidden files and system files
            if file_path.name.startswith('.') or file_path.name.startswith('~'):
                continue
            if file_path.name in ['Thumbs.db', 'desktop.ini', '.DS_Store']:
                continue
            
            # Check if file exists and is readable
            try:
                if file_path.is_file() and os.access(file_path, os.R_OK):
                    filtered_files.append(file_path)
            except (OSError, PermissionError):
                logger.warning(f"Permission denied: {file_path}")
        
        # Sort by modification time (oldest first)
        filtered_files.sort(key=lambda x: x.stat().st_mtime)
        
        return filtered_files
    
    async def _process_batch(self, file_paths: List[Path], session: Session):
        """Process a batch of files"""
        tasks = []
        for file_path in file_paths:
            task = self._process_single_file(file_path, session)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error processing file: {result}")
                self.stats['errors'] += 1
            elif result:
                self._update_stats(result)
    
    async def _process_single_file(self, file_path: Path, session: Session) -> Optional[FileInfo]:
        """Process a single file"""

        # EARLY EXIT — file already in DB
        existing_path = session.query(File).filter(
            File.file_path == str(file_path)
        ).first()

        if existing_path:
            logger.info(f"File already exists in DB, skipping: {file_path}")

            stat = file_path.stat()
            return FileInfo(
                path=file_path,
                size=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime),
                hash="",
                is_duplicate=True,
                duplicate_of=existing_path.file_path
            )

        try:
            # Build file info
            stat = file_path.stat()
            file_info = FileInfo(
                path=file_path,
                size=stat.st_size,
                mtime=datetime.fromtimestamp(stat.st_mtime),
                hash=""
            )

            # Compute hashes
            file_info.hash = await self.duplicate_detector.compute_file_hash(file_path)
            file_info.perceptual_hash = await self.duplicate_detector.compute_perceptual_hash(file_path)

            # Check duplicates
            duplicates = await self.duplicate_detector.find_duplicates(
                file_path, file_info.hash, file_info.perceptual_hash
            )

            # Unique dedupe collapse
            if duplicates:
                seen_ids = set()
                unique = []
                for d in duplicates:
                    if d[0] not in seen_ids:
                        seen_ids.add(d[0])
                        unique.append(d)
                duplicates = unique

            if duplicates:
                file_info.is_duplicate = True
                file_info.duplicate_of = duplicates[0][1]
                logger.info(f"Duplicate found: {file_path} -> {file_info.duplicate_of}")
                return file_info

            # Otherwise ingest
            await self._ingest_file(file_info, session)
            return file_info

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return FileInfo(
                path=file_path,
                size=0,
                mtime=datetime.now(),
                hash="",
                error=str(e)
            )

    
    async def _ingest_file(self, file_info: FileInfo, session: Session):
        """Ingest a non-duplicate file"""
        # Determine file type
        file_type = get_file_type(file_info.path)
        mime_type = self._guess_mime_type(file_info.path)
        
        # Create database record
        file_record = File(
            file_path=str(file_info.path),
            file_hash=file_info.hash,
            perceptual_hash=file_info.perceptual_hash,
            file_size=file_info.size,
            file_type=file_type,
            mime_type=mime_type,
            ingested_at=datetime.utcnow(),
            processed=False
        )
        
        session.add(file_record)
        session.flush()  # Get the ID
        
        # Add to processing queue
        queue_item = ProcessingQueue(
            file_id=file_record.id,
            task_type='analyze',
            priority=0,
            status='pending'
        )
        session.add(queue_item)
        
        # Copy to organized location if configured
        if self.config.paths.output:
            await self._organize_file(file_info, file_record.id, session)
        
        logger.info(f"Ingested file: {file_info.path} (ID: {file_record.id})")
    
    def _store_duplicate_info(self, *args, **kwargs):
        # PATCH C — Disabled. We no longer store duplicates in DB.
        return



    
    async def _organize_file(self, file_info: FileInfo, file_id: int, session: Session):
        """Organize file into output directory"""
        output_dir = Path(self.config.paths.output)
        
        # Create organization path based on date and type
        organized_path = await self._create_organization_path(file_info, output_dir)
        
        # Ensure directory exists
        organized_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy/hardlink file
        if await self.duplicate_detector.create_hardlink(file_info.path, organized_path):
            logger.info(f"Organized: {file_info.path} -> {organized_path}")
            
            # Update database with organized path
            # (We would store this in a separate table or field)
            return organized_path
        
        return None
    
    async def _create_organization_path(self, file_info: FileInfo, base_dir: Path) -> Path:
        """Create organized path structure"""
        # Default structure: Year/Month/Day/Filename
        date_str = file_info.mtime.strftime("%Y/%m/%d")
        
        # Add file type subdirectory
        file_type = get_file_type(file_info.path)
        type_dir = file_type.capitalize() + "s"  # Photos, Videos, Audio
        
        # Create unique filename
        filename = file_info.path.name
        counter = 1
        
        while True:
            organized_path = base_dir / type_dir / date_str / filename
            if not organized_path.exists():
                break
            
            # Add counter to filename
            stem = file_info.path.stem
            suffix = file_info.path.suffix
            filename = f"{stem}_{counter}{suffix}"
            counter += 1
        
        return organized_path
    
    def _guess_mime_type(self, file_path: Path) -> str:
        """Guess MIME type from file extension"""
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.tiff': 'image/tiff',
            '.webp': 'image/webp',
            '.heic': 'image/heic',
            '.mp4': 'video/mp4',
            '.mov': 'video/quicktime',
            '.avi': 'video/x-msvideo',
            '.mkv': 'video/x-matroska',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.flac': 'audio/flac',
        }
        
        ext = file_path.suffix.lower()
        return mime_types.get(ext, 'application/octet-stream')
    
    def _update_stats(self, file_info: FileInfo):
        """Update ingestion statistics"""
        if file_info.error:
            self.stats['errors'] += 1
        elif file_info.is_duplicate:
            self.stats['duplicates_skipped'] += 1
        else:
            self.stats['ingested'] += 1
            self.stats['total_size'] += file_info.size
    
    async def cleanup_orphans(self, directory: Path):
        """Clean up orphaned files (in database but not on disk)"""
        with self.db_manager.get_session() as session:
            # Get all files in database for this directory
            db_files = session.query(File).filter(
                File.file_path.like(f"{directory}%")
            ).all()
            
            orphans = []
            for db_file in db_files:
                if not Path(db_file.file_path).exists():
                    orphans.append(db_file)
            
            logger.info(f"Found {len(orphans)} orphaned files")
            
            # Mark orphans as deleted
            for orphan in orphans:
                orphan.error = "File not found on disk"
                orphan.processed = True
            
            session.commit()
            return len(orphans)