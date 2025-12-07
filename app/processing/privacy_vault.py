# app/processing/privacy_vault.py

"""
Privacy vault for sensitive content detection and management
"""
import asyncio
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import shutil
import json
from typing import Any
from sqlalchemy.orm import Session

from app.db.models import File, AnalysisResult
from app.core.ollama_client import OllamaClient
from app.core.intelligence_engine import IntelligenceEngine

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of sensitivity detection"""
    file_path: Path
    is_sensitive: bool
    category: str  # 'document', 'nudity', 'personal', 'financial', etc.
    confidence: float
    details: Dict
    action_taken: Optional[str] = None


class SensitivityDetector:
    """Detect sensitive content in files"""
    
    def __init__(self, config: Dict, db_session: Session):
        self.config = config
        self.db = db_session
        self.ollama = OllamaClient()
        self.intelligence = IntelligenceEngine(config)
        
        # Sensitivity thresholds
        self.thresholds = {
            'document': 0.7,
            'nudity': 0.8,
            'financial': 0.6,
            'personal_id': 0.9,
            'medical': 0.7
        }
    
    async def detect_sensitive_content(self, file_path: Path) -> SensitivityResult:
        """Detect sensitive content in file"""
        # Default result
        result = SensitivityResult(
            file_path=file_path,
            is_sensitive=False,
            category='safe',
            confidence=0.0,
            details={}
        )
        
        try:
            # Run AI analysis
            analysis = await self.intelligence.analyze_media(file_path)
            
            if analysis.image_analysis:
                # Check if AI marked as sensitive
                if analysis.image_analysis.is_sensitive:
                    result.is_sensitive = True
                    result.category = 'ai_flagged'
                    result.confidence = analysis.image_analysis.confidence
                    result.details = {'reason': 'AI flagged as sensitive'}
                    return result
                
                # Check for text content (documents)
                if analysis.image_analysis.contains_text:
                    text_analysis = await self._analyze_text_content(file_path)
                    if text_analysis['is_sensitive']:
                        result.is_sensitive = True
                        result.category = text_analysis['category']
                        result.confidence = text_analysis['confidence']
                        result.details = text_analysis['details']
                        return result
            
            # Check file patterns
            pattern_result = await self._check_file_patterns(file_path)
            if pattern_result['is_sensitive']:
                result.is_sensitive = True
                result.category = pattern_result['category']
                result.confidence = pattern_result['confidence']
                result.details = pattern_result['details']
                return result
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting sensitive content in {file_path}: {e}")
            result.details = {'error': str(e)}
            return result
    
    async def _analyze_text_content(self, file_path: Path) -> Dict:
        """Analyze text content for sensitive information"""
        try:
            # Use OCR to extract text (simplified)
            import pytesseract
            from PIL import Image
            import re
            
            with Image.open(file_path) as img:
                text = pytesseract.image_to_string(img)
            
            # Check for sensitive patterns
            patterns = {
                'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
                'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
                'phone': r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                'medical': r'\b(patient|diagnosis|prescription|medical|health)\b',
                'financial': r'\b(account|routing|balance|transaction|statement)\b',
            }
            
            for category, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    return {
                        'is_sensitive': True,
                        'category': category,
                        'confidence': min(0.9, len(matches) * 0.3),
                        'details': {
                            'pattern': category,
                            'matches': len(matches),
                            'sample': matches[0] if matches else ''
                        }
                    }
            
            # Check for document-like structure
            lines = text.strip().split('\n')
            if len(lines) > 5:
                # Looks like a document
                return {
                    'is_sensitive': True,
                    'category': 'document',
                    'confidence': 0.7,
                    'details': {'line_count': len(lines)}
                }
            
            return {'is_sensitive': False, 'category': 'safe', 'confidence': 0.0, 'details': {}}
            
        except Exception as e:
            logger.error(f"Error analyzing text content: {e}")
            return {'is_sensitive': False, 'category': 'error', 'confidence': 0.0, 'details': {'error': str(e)}}
    
    async def _check_file_patterns(self, file_path: Path) -> Dict:
        """Check file name and path for sensitive patterns"""
        filename = file_path.name.lower()
        
        sensitive_keywords = {
            'document': ['document', 'contract', 'agreement', 'invoice', 'receipt', 'tax', 'bank'],
            'id': ['passport', 'license', 'id_card', 'ssn', 'social_security'],
            'medical': ['medical', 'health', 'prescription', 'lab', 'test', 'report'],
            'financial': ['financial', 'statement', 'account', 'transaction', 'credit'],
            'personal': ['private', 'confidential', 'secret', 'personal']
        }
        
        for category, keywords in sensitive_keywords.items():
            for keyword in keywords:
                if keyword in filename:
                    return {
                        'is_sensitive': True,
                        'category': category,
                        'confidence': 0.8,
                        'details': {'keyword': keyword}
                    }
        
        return {'is_sensitive': False, 'category': 'safe', 'confidence': 0.0, 'details': {}}
    
    async def batch_detect(self, file_paths: List[Path]) -> List[SensitivityResult]:
        """Detect sensitive content in multiple files"""
        tasks = [self.detect_sensitive_content(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch detection error: {result}")
            else:
                valid_results.append(result)
        
        return valid_results


class PrivacyVault:
    """Manage privacy vault for sensitive files"""
    
    def __init__(self, config: Dict, db_session: Session):
        self.config = config
        self.db = db_session
        self.detector = SensitivityDetector(config, db_session)
        
        # Vault configuration
        self.vault_path = Path(config.get('paths', {}).get('vault', './vault'))
        self.encrypted = config.get('vault', {}).get('encrypted', False)
        
        # Create vault structure
        self._init_vault()
    
    def _init_vault(self):
        """Initialize vault directory structure"""
        self.vault_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        categories = ['documents', 'personal', 'financial', 'medical', 'other']
        for category in categories:
            (self.vault_path / category).mkdir(exist_ok=True)
        
        # Create index file
        index_file = self.vault_path / 'index.json'
        if not index_file.exists():
            with open(index_file, 'w') as f:
                json.dump({'version': '1.0', 'created': datetime.now().isoformat()}, f)
    
    async def add_to_vault(self, file_path: Path, 
                          sensitivity_result: SensitivityResult) -> Tuple[bool, Optional[Path]]:
        """Add file to privacy vault"""
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False, None
            
            # Create vault path
            vault_category = self.vault_path / sensitivity_result.category
            vault_filename = self._generate_vault_filename(file_path, sensitivity_result)
            vault_path = vault_category / vault_filename
            
            # Ensure unique filename
            counter = 1
            while vault_path.exists():
                vault_path = vault_category / f"{file_path.stem}_{counter}{file_path.suffix}"
                counter += 1
            
            # Move file to vault
            if self.encrypted:
                encrypted_path = await self._encrypt_file(file_path, vault_path)
                if encrypted_path:
                    # Remove original
                    file_path.unlink()
                    vault_path = encrypted_path
                else:
                    # Fallback to copy
                    shutil.copy2(file_path, vault_path)
            else:
                shutil.move(str(file_path), str(vault_path))
            
            # Update database
            await self._update_database(file_path, vault_path, sensitivity_result)
            
            # Update vault index
            await self._update_vault_index(vault_path, sensitivity_result)
            
            logger.info(f"Added to vault: {file_path} -> {vault_path}")
            return True, vault_path
            
        except Exception as e:
            logger.error(f"Error adding {file_path} to vault: {e}")
            return False, None
    
    def _generate_vault_filename(self, file_path: Path, 
                               sensitivity_result: SensitivityResult) -> str:
        """Generate secure filename for vault"""
        # Create hash-based filename
        file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return f"{timestamp}_{file_hash}{file_path.suffix}"
    
    async def _encrypt_file(self, source: Path, target: Path) -> Optional[Path]:
        """Encrypt file before storing in vault"""
        try:
            # Use cryptography library if available
            try:
                from cryptography.fernet import Fernet
                from cryptography.hazmat.primitives import hashes
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                import base64
                import os
                
                # Generate or load key
                key_file = self.vault_path / '.vault_key'
                if key_file.exists():
                    with open(key_file, 'rb') as f:
                        key = f.read()
                else:
                    # Generate new key
                    password = os.urandom(32)
                    salt = os.urandom(16)
                    kdf = PBKDF2HMAC(
                        algorithm=hashes.SHA256(),
                        length=32,
                        salt=salt,
                        iterations=100000,
                    )
                    key = base64.urlsafe_b64encode(kdf.derive(password))
                    
                    # Save key
                    with open(key_file, 'wb') as f:
                        f.write(key)
                
                # Encrypt file
                fernet = Fernet(key)
                
                with open(source, 'rb') as f:
                    original_data = f.read()
                
                encrypted_data = fernet.encrypt(original_data)
                
                # Save encrypted
                encrypted_path = target.with_suffix(target.suffix + '.enc')
                with open(encrypted_path, 'wb') as f:
                    f.write(encrypted_data)
                
                return encrypted_path
                
            except ImportError:
                logger.warning("cryptography not installed, skipping encryption")
                return None
                
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            return None
    
    async def _update_database(self, original_path: Path, vault_path: Path,
                             sensitivity_result: SensitivityResult):
        """Update database with vault information"""
        # Find file record
        file_record = self.db.query(File).filter(
            File.file_path == str(original_path)
        ).first()
        
        if file_record:
            # Update record
            file_record.file_path = str(vault_path)
            file_record.error = f"Vaulted: {sensitivity_result.category}"
            file_record.processed = True
            
            # Create vault record (you'd have a separate table for this)
            self.db.commit()
    
    async def _update_vault_index(self, vault_path: Path, 
                                sensitivity_result: SensitivityResult):
        """Update vault index file"""
        index_file = self.vault_path / 'index.json'
        
        try:
            if index_file.exists():
                with open(index_file, 'r') as f:
                    index = json.load(f)
            else:
                index = {'entries': []}
            
            # Add entry
            entry = {
                'original_path': str(sensitivity_result.file_path),
                'vault_path': str(vault_path),
                'category': sensitivity_result.category,
                'confidence': sensitivity_result.confidence,
                'added': datetime.now().isoformat(),
                'details': sensitivity_result.details
            }
            
            if 'entries' not in index:
                index['entries'] = []
            
            index['entries'].append(entry)
            
            # Write back
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating vault index: {e}")
    
    async def scan_and_vault(self, directory: Path, 
                           recursive: bool = True) -> Dict[str, Any]:
        """Scan directory and vault sensitive files"""
        # Find all files
        media_extensions = {'.jpg', '.jpeg', '.png', '.pdf', '.doc', '.docx'}
        files = []
        
        if recursive:
            for ext in media_extensions:
                files.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in media_extensions:
                files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Scanning {len(files)} files for sensitive content")
        
        results = {
            'scanned': len(files),
            'sensitive': 0,
            'vaulted': 0,
            'errors': 0,
            'details': []
        }
        
        for file_path in files:
            try:
                # Detect sensitivity
                sensitivity = await self.detector.detect_sensitive_content(file_path)
                
                if sensitivity.is_sensitive:
                    results['sensitive'] += 1
                    
                    # Add to vault
                    success, vault_path = await self.add_to_vault(file_path, sensitivity)
                    
                    if success:
                        results['vaulted'] += 1
                        sensitivity.action_taken = f"vaulted to {vault_path}"
                    else:
                        results['errors'] += 1
                        sensitivity.action_taken = 'failed to vault'
                
                results['details'].append({
                    'file': str(file_path),
                    'sensitive': sensitivity.is_sensitive,
                    'category': sensitivity.category,
                    'confidence': sensitivity.confidence,
                    'action': sensitivity.action_taken
                })
                
            except Exception as e:
                results['errors'] += 1
                logger.error(f"Error scanning {file_path}: {e}")
        
        return results
    
    async def list_vault_contents(self) -> List[Dict]:
        """List all files in vault"""
        index_file = self.vault_path / 'index.json'
        
        if not index_file.exists():
            return []
        
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            return index.get('entries', [])
        except Exception as e:
            logger.error(f"Error reading vault index: {e}")
            return []
    
    async def restore_from_vault(self, vault_entry_id: int, 
                               restore_path: Optional[Path] = None) -> bool:
        """Restore file from vault"""
        entries = await self.list_vault_contents()
        
        if vault_entry_id < 0 or vault_entry_id >= len(entries):
            logger.error(f"Invalid vault entry ID: {vault_entry_id}")
            return False
        
        entry = entries[vault_entry_id]
        vault_path = Path(entry['vault_path'])
        
        if not vault_path.exists():
            logger.error(f"Vault file not found: {vault_path}")
            return False
        
        try:
            # Determine restore path
            if restore_path is None:
                original = Path(entry['original_path'])
                restore_path = original.parent / f"restored_{original.name}"
            
            # Decrypt if encrypted
            if vault_path.suffix == '.enc':
                decrypted = await self._decrypt_file(vault_path, restore_path)
                if not decrypted:
                    return False
            else:
                shutil.copy2(vault_path, restore_path)
            
            # Update database
            await self._update_restore_database(entry, restore_path)
            
            # Remove from vault index
            await self._remove_from_vault_index(vault_entry_id)
            
            logger.info(f"Restored from vault: {vault_path} -> {restore_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring from vault: {e}")
            return False
    
    async def _decrypt_file(self, encrypted_path: Path, target_path: Path) -> bool:
        """Decrypt vault file"""
        try:
            from cryptography.fernet import Fernet
            import base64
            
            # Load key
            key_file = self.vault_path / '.vault_key'
            if not key_file.exists():
                logger.error("Vault key not found")
                return False
            
            with open(key_file, 'rb') as f:
                key = f.read()
            
            # Decrypt
            fernet = Fernet(key)
            
            with open(encrypted_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = fernet.decrypt(encrypted_data)
            
            # Save decrypted
            with open(target_path, 'wb') as f:
                f.write(decrypted_data)
            
            return True
            
        except ImportError:
            logger.error("cryptography not installed")
            return False
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return False
    
    async def _update_restore_database(self, entry: Dict, restore_path: Path):
        """Update database when file is restored"""
        # Find original record (might be marked as vaulted)
        file_record = self.db.query(File).filter(
            File.file_path == entry['vault_path']
        ).first()
        
        if file_record:
            file_record.file_path = str(restore_path)
            file_record.error = None
            file_record.processed = False
            self.db.commit()
    
    async def _remove_from_vault_index(self, entry_id: int):
        """Remove entry from vault index"""
        index_file = self.vault_path / 'index.json'
        
        if not index_file.exists():
            return
        
        try:
            with open(index_file, 'r') as f:
                index = json.load(f)
            
            if 'entries' in index and 0 <= entry_id < len(index['entries']):
                del index['entries'][entry_id]
                
                with open(index_file, 'w') as f:
                    json.dump(index, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error removing from vault index: {e}")