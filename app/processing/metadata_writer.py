# app/processing/metadata_writer.py

"""
Write AI-generated metadata back to files
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from sqlalchemy.orm import Session

from app.db.models import File, AnalysisResult

logger = logging.getLogger(__name__)


class EXIFUpdater:
    """Update EXIF and other metadata in files"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.tiff', '.tif', '.png', '.heic'}
    
    def __init__(self):
        self.exiftool_available = self._check_exiftool()
    
    def _check_exiftool(self) -> bool:
        """Check if exiftool is available"""
        try:
            result = subprocess.run(['exiftool', '-ver'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            logger.warning("exiftool not found in PATH")
            return False
    
    async def write_metadata(self, file_path: Path, metadata: Dict) -> bool:
        """Write metadata to file using exiftool"""
        if not self.exiftool_available:
            logger.error("exiftool not available")
            return False
        
        if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported format for metadata writing: {file_path.suffix}")
            return False
        
        try:
            # Prepare exiftool command
            cmd = ['exiftool', '-overwrite_original', '-charset', 'utf8']
            
            # Add metadata tags
            for key, value in metadata.items():
                if value is not None:
                    tag = self._map_key_to_exif_tag(key)
                    if tag:
                        cmd.extend([f'-{tag}={str(value)}'])
            
            cmd.append(str(file_path))
            
            # Run exiftool
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info(f"Updated metadata for {file_path}")
                return True
            else:
                logger.error(f"exiftool error for {file_path}: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Error writing metadata to {file_path}: {e}")
            return False
    
    def _map_key_to_exif_tag(self, key: str) -> Optional[str]:
        """Map metadata key to exiftool tag name"""
        tag_map = {
            'caption': 'XPComment',  # or 'Description', 'UserComment'
            'tags': 'Keywords',
            'rating': 'Rating',
            'creator': 'Artist',
            'copyright': 'Copyright',
            'title': 'Title',
            'subject': 'Subject',
            'location': 'Location',
            'city': 'City',
            'country': 'Country',
            'keywords': 'Keywords',
            'description': 'Description',
            'comment': 'UserComment',
        }
        
        return tag_map.get(key.lower())
    
    async def write_iptc(self, file_path: Path, metadata: Dict) -> bool:
        """Write IPTC metadata specifically"""
        if not self.exiftool_available:
            return False
        
        try:
            cmd = ['exiftool', '-overwrite_original']
            
            # IPTC tags
            if 'caption' in metadata:
                cmd.extend([f'-IPTC:Caption-Abstract={metadata["caption"]}'])
            
            if 'keywords' in metadata:
                if isinstance(metadata['keywords'], list):
                    for keyword in metadata['keywords']:
                        cmd.extend([f'-IPTC:Keywords={keyword}'])
                else:
                    cmd.extend([f'-IPTC:Keywords={metadata["keywords"]}'])
            
            if 'title' in metadata:
                cmd.extend([f'-IPTC:ObjectName={metadata["title"]}'])
            
            cmd.append(str(file_path))
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error writing IPTC to {file_path}: {e}")
            return False
    
    async def write_xmp(self, file_path: Path, metadata: Dict) -> bool:
        """Write XMP metadata (more flexible)"""
        if not self.exiftool_available:
            return False
        
        try:
            cmd = ['exiftool', '-overwrite_original']
            
            # XMP tags for AI-generated content
            if 'caption' in metadata:
                cmd.extend([
                    f'-XMP:Description={metadata["caption"]}',
                    f'-XMP:UserComment={metadata["caption"]}'
                ])
            
            if 'tags' in metadata:
                if isinstance(metadata['tags'], list):
                    tags_str = ', '.join(metadata['tags'])
                    cmd.extend([f'-XMP:Subject={tags_str}'])
            
            if 'ai_generated' in metadata:
                cmd.extend([f'-XMP:CreatorTool=Mnemosyne AI'])
            
            cmd.append(str(file_path))
            
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error writing XMP to {file_path}: {e}")
            return False


class MetadataWriter:
    """Write AI-generated metadata back to files"""
    
    def __init__(self, config: Dict, db_session: Session):
        self.config = config
        self.db = db_session
        self.exif_updater = EXIFUpdater()
        
    async def write_file_metadata(self, file_id: int) -> bool:
        """Write AI-generated metadata to file"""
        # Get file and analysis
        file_record = self.db.query(File).filter(File.id == file_id).first()
        if not file_record:
            logger.error(f"File not found: ID {file_id}")
            return False
        
        analysis = self.db.query(AnalysisResult).filter(
            AnalysisResult.file_id == file_id
        ).first()
        
        if not analysis:
            logger.warning(f"No analysis found for file ID {file_id}")
            return False
        
        file_path = Path(file_record.file_path)
        if not file_path.exists():
            logger.error(f"File not on disk: {file_path}")
            return False
        
        # Prepare metadata
        metadata = await self._prepare_metadata(analysis)
        
        # Write metadata
        success = await self.exif_updater.write_metadata(file_path, metadata)
        
        if success:
            logger.info(f"Written metadata to {file_path}")
            # Update database
            file_record.processed = True
            self.db.commit()
        
        return success
    
    async def _prepare_metadata(self, analysis: AnalysisResult) -> Dict:
        """Prepare metadata dictionary from analysis"""
        metadata = {}
        
        if analysis.caption:
            metadata['caption'] = analysis.caption[:1000]  # Limit length
        
        if analysis.tags:
            try:
                tags = json.loads(analysis.tags)
                if isinstance(tags, list):
                    metadata['tags'] = tags
                    metadata['keywords'] = ', '.join(tags[:20])  # Limit keywords
            except:
                pass
        
        if analysis.mood:
            metadata['comment'] = f"Mood: {analysis.mood}"
        
        if analysis.aesthetic_score:
            rating = int(analysis.aesthetic_score / 2)  # Convert 0-10 to 0-5
            metadata['rating'] = min(5, max(0, rating))
        
        # Add AI attribution
        metadata['ai_generated'] = True
        metadata['creator'] = 'Mnemosyne AI'
        metadata['software'] = 'Mnemosyne Digital Life Archival System'
        
        return metadata
    
    async def batch_write_metadata(self, file_ids: List[int]) -> Dict[str, Any]:
        """Write metadata for multiple files"""
        results = {
            'total': len(file_ids),
            'success': 0,
            'failed': 0,
            'errors': []
        }
        
        for file_id in file_ids:
            try:
                success = await self.write_file_metadata(file_id)
                if success:
                    results['success'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Failed to write metadata for file ID {file_id}")
            except Exception as e:
                results['failed'] += 1
                results['errors'].append(f"Error processing file ID {file_id}: {str(e)}")
        
        return results
    
    async def write_all_pending(self) -> Dict[str, Any]:
        """Write metadata for all analyzed but not written files"""
        # Find analyzed files without metadata written
        analyzed_files = self.db.query(File).join(AnalysisResult).filter(
            File.processed == False
        ).all()
        
        file_ids = [f.id for f in analyzed_files]
        
        logger.info(f"Found {len(file_ids)} files pending metadata write")
        return await self.batch_write_metadata(file_ids)
    
    async def verify_metadata(self, file_path: Path) -> Dict:
        """Verify metadata was written correctly"""
        if not self.exif_updater.exiftool_available:
            return {'error': 'exiftool not available'}
        
        try:
            cmd = ['exiftool', '-j', str(file_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                metadata = json.loads(result.stdout)[0]
                
                # Check for our metadata
                checks = {
                    'has_description': 'Description' in metadata or 'XPComment' in metadata,
                    'has_keywords': 'Keywords' in metadata,
                    'has_rating': 'Rating' in metadata,
                    'has_software': 'Software' in metadata and 'Mnemosyne' in metadata['Software']
                }
                
                return {
                    'success': True,
                    'checks': checks,
                    'metadata': {k: v for k, v in metadata.items() 
                               if k in ['Description', 'Keywords', 'Rating', 'Software', 'Artist', 'Copyright']}
                }
            else:
                return {'error': result.stderr}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def restore_backup(self, file_path: Path) -> bool:
        """Restore original file from backup (exiftool creates *_original)"""
        backup_path = file_path.with_suffix(file_path.suffix + '_original')
        
        if backup_path.exists():
            try:
                backup_path.replace(file_path)
                logger.info(f"Restored original for {file_path}")
                return True
            except Exception as e:
                logger.error(f"Error restoring backup for {file_path}: {e}")
                return False
        
        return False