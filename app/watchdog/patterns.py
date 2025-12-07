# app/watchdog/patterns.py

"""
Pattern matching and filtering for file system events
"""
import fnmatch
import re
import logging
from pathlib import Path
from typing import List, Pattern, Optional, Set, Tuple, Any, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PatternRule:
    """Pattern matching rule"""
    pattern: str
    is_regex: bool = False
    case_sensitive: bool = False
    match_directories: bool = True
    match_files: bool = True
    
    def __post_init__(self):
        # Compile regex if needed
        if self.is_regex:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            try:
                self.compiled_pattern = re.compile(self.pattern, flags)
            except re.error as e:
                logger.error(f"Invalid regex pattern '{self.pattern}': {e}")
                # Fallback to literal match
                self.is_regex = False
    
    def matches(self, path: Path) -> bool:
        """
        Check if path matches pattern
        
        Args:
            path: Path to check
            
        Returns:
            True if path matches pattern
        """
        # Check if we should match this type
        if path.is_dir() and not self.match_directories:
            return False
        if path.is_file() and not self.match_files:
            return False
        
        # Get string to match
        path_str = str(path)
        if not self.case_sensitive:
            path_str = path_str.lower()
            pattern = self.pattern.lower() if not self.is_regex else self.pattern
        else:
            pattern = self.pattern
        
        # Match using appropriate method
        if self.is_regex:
            return bool(self.compiled_pattern.search(path_str))
        else:
            return fnmatch.fnmatch(path_str, pattern)


class PatternFilter:
    """
    Filter files and directories based on patterns
    """
    
    def __init__(self, ignore_patterns: List[str] = None, 
                 ignore_directories: List[str] = None):
        """
        Initialize pattern filter
        
        Args:
            ignore_patterns: List of patterns to ignore
            ignore_directories: List of directory patterns to ignore
        """
        self.ignore_patterns = ignore_patterns or self._get_default_ignore_patterns()
        self.ignore_directories = ignore_directories or self._get_default_ignore_directories()
        
        # Compile rules
        self.ignore_rules = self._compile_rules()
        
        # Cache for performance
        self.cache: Dict[Path, bool] = {}
        self.cache_max_size = 10000
        
        logger.info(f"PatternFilter initialized with {len(self.ignore_rules)} rules")
    
    def _get_default_ignore_patterns(self) -> List[str]:
        """Get default ignore patterns"""
        return [
            # Hidden files and directories
            '.*',
            '*/.*',
            
            # System files
            '*.tmp', '*.temp', '*.bak', '*.backup',
            'Thumbs.db', 'desktop.ini', '.DS_Store',
            '*.lnk', '*.url',
            
            # Temporary files
            '~*',
            '*.swp', '*.swo',
            
            # Cache and thumbnail files
            '*.cache', '*.thumb', 'thumb*.db',
            
            # Application-specific
            '._*',  # macOS resource fork
            '.Spotlight-*', '.Trashes', '.fseventsd',
            
            # Network and synchronization
            '.dropbox', '.dropbox.cache',
            '.git', '.svn', '.hg',
            
            # Virtual and mounted filesystems
            '.gvfs', '.mtp', '.mtp+',
        ]
    
    def _get_default_ignore_directories(self) -> List[str]:
        """Get default ignore directories"""
        return [
            '@eaDir',  # Synology
            '.thumbnails', '.thumbnail',
            '.trash', '.Trash', '.Trash-*',
            '.stfolder',  # Syncthing
            '.syncthing', '.syncthing.*',
            'lost+found',
            'System Volume Information',
            '$RECYCLE.BIN',
            '.TemporaryItems',
            '.DocumentRevisions-V100',
        ]
    
    def _compile_rules(self) -> List[PatternRule]:
        """Compile pattern strings into PatternRule objects"""
        rules = []
        
        # Add file patterns
        for pattern in self.ignore_patterns:
            try:
                rule = PatternRule(
                    pattern=pattern,
                    is_regex=self._is_regex_pattern(pattern),
                    case_sensitive=False,
                    match_directories=True,
                    match_files=True
                )
                rules.append(rule)
            except Exception as e:
                logger.error(f"Failed to compile pattern '{pattern}': {e}")
        
        # Add directory patterns
        for dir_pattern in self.ignore_directories:
            try:
                rule = PatternRule(
                    pattern=dir_pattern,
                    is_regex=self._is_regex_pattern(dir_pattern),
                    case_sensitive=False,
                    match_directories=True,
                    match_files=False  # Only match directories
                )
                rules.append(rule)
            except Exception as e:
                logger.error(f"Failed to compile directory pattern '{dir_pattern}': {e}")
        
        return rules
    
    def _is_regex_pattern(self, pattern: str) -> bool:
        """
        Check if pattern looks like a regex
        
        Args:
            pattern: Pattern string
            
        Returns:
            True if pattern appears to be regex
        """
        # Simple heuristic: regex patterns often contain special characters
        regex_chars = {'^', '$', '(', ')', '[', ']', '{', '}', '|', '?', '+', '\\'}
        return any(char in pattern for char in regex_chars)
    
    def should_ignore(self, path: Path) -> bool:
        """
        Check if path should be ignored
        
        Args:
            path: Path to check
            
        Returns:
            True if path should be ignored
        """
        # Check cache first
        if path in self.cache:
            return self.cache[path]
        
        # Check if path exists
        if not path.exists():
            # Non-existent paths are ignored
            self._update_cache(path, True)
            return True
        
        # Check all rules
        for rule in self.ignore_rules:
            if rule.matches(path):
                logger.debug(f"Ignoring {path} (matched pattern: {rule.pattern})")
                self._update_cache(path, True)
                return True
        
        # Also check parent directories
        parent_ignored = self._check_parent_directories(path)
        if parent_ignored:
            self._update_cache(path, True)
            return True
        
        # Path should not be ignored
        self._update_cache(path, False)
        return False
    
    def _check_parent_directories(self, path: Path) -> bool:
        """
        Check if any parent directory should be ignored
        
        Args:
            path: Path to check
            
        Returns:
            True if any parent directory should be ignored
        """
        # Check each parent directory
        for parent in path.parents:
            # Check cache first
            if parent in self.cache:
                if self.cache[parent]:
                    return True
                continue
            
            # Check rules for this directory
            for rule in self.ignore_rules:
                if rule.match_directories and rule.matches(parent):
                    self._update_cache(parent, True)
                    return True
            
            # If directory passed all rules, cache it
            self._update_cache(parent, False)
        
        return False
    
    def _update_cache(self, path: Path, should_ignore: bool):
        """
        Update cache with path decision
        
        Args:
            path: Path to cache
            should_ignore: Whether path should be ignored
        """
        # Limit cache size
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entries (first 10%)
            remove_count = self.cache_max_size // 10
            keys_to_remove = list(self.cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self.cache[key]
        
        self.cache[path] = should_ignore
    
    def add_pattern(self, pattern: str, is_directory: bool = False):
        """
        Add a new pattern to filter
        
        Args:
            pattern: Pattern string
            is_directory: Whether pattern applies only to directories
        """
        try:
            rule = PatternRule(
                pattern=pattern,
                is_regex=self._is_regex_pattern(pattern),
                case_sensitive=False,
                match_directories=is_directory,
                match_files=not is_directory
            )
            
            self.ignore_rules.append(rule)
            
            if is_directory:
                self.ignore_directories.append(pattern)
            else:
                self.ignore_patterns.append(pattern)
            
            # Clear cache since rules changed
            self.cache.clear()
            
            logger.info(f"Added pattern: {pattern} (directory: {is_directory})")
            
        except Exception as e:
            logger.error(f"Failed to add pattern '{pattern}': {e}")
    
    def remove_pattern(self, pattern: str):
        """
        Remove a pattern from filter
        
        Args:
            pattern: Pattern string to remove
        """
        # Remove from rules
        self.ignore_rules = [
            rule for rule in self.ignore_rules 
            if rule.pattern != pattern
        ]
        
        # Remove from pattern lists
        if pattern in self.ignore_patterns:
            self.ignore_patterns.remove(pattern)
        
        if pattern in self.ignore_directories:
            self.ignore_directories.remove(pattern)
        
        # Clear cache
        self.cache.clear()
        
        logger.info(f"Removed pattern: {pattern}")
    
    def test_pattern(self, path: Path, pattern: str) -> bool:
        """
        Test if path matches a specific pattern
        
        Args:
            path: Path to test
            pattern: Pattern to test against
            
        Returns:
            True if path matches pattern
        """
        try:
            rule = PatternRule(
                pattern=pattern,
                is_regex=self._is_regex_pattern(pattern),
                case_sensitive=False,
                match_directories=True,
                match_files=True
            )
            
            return rule.matches(path)
            
        except Exception as e:
            logger.error(f"Error testing pattern '{pattern}': {e}")
            return False
    
    def get_matching_patterns(self, path: Path) -> List[str]:
        """
        Get all patterns that match a path
        
        Args:
            path: Path to check
            
        Returns:
            List of matching pattern strings
        """
        matching_patterns = []
        
        for rule in self.ignore_rules:
            if rule.matches(path):
                matching_patterns.append(rule.pattern)
        
        return matching_patterns
    
    def get_stats(self) -> dict:
        """Get filter statistics"""
        return {
            'total_rules': len(self.ignore_rules),
            'file_patterns': len(self.ignore_patterns),
            'directory_patterns': len(self.ignore_directories),
            'cache_size': len(self.cache),
            'cache_max_size': self.cache_max_size,
        }


class FilePatternMatcher:
    """
    Advanced pattern matching for file types and categories
    """
    
    def __init__(self):
        """Initialize pattern matcher"""
        # File type patterns
        self.file_type_patterns = {
            'image': [
                '*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.tif',
                '*.webp', '*.heic', '*.heif', '*.raw', '*.nef', '*.cr2', '*.arw',
                '*.dng', '*.orf', '*.sr2', '*.raf', '*.rw2', '*.pef', '*.srw',
            ],
            'video': [
                '*.mp4', '*.mov', '*.avi', '*.mkv', '*.webm', '*.flv', '*.wmv',
                '*.m4v', '*.mpg', '*.mpeg', '*.3gp', '*.mts', '*.m2ts', '*.ts',
                '*.vob', '*.ogv', '*.rm', '*.rmvb', '*.asf', '*.f4v',
            ],
            'audio': [
                '*.mp3', '*.wav', '*.flac', '*.m4a', '*.aac', '*.ogg', '*.wma',
                '*.opus', '*.ape', '*.alac', '*.aiff', '*.mid', '*.midi', '*.amr',
            ],
            'document': [
                '*.pdf', '*.doc', '*.docx', '*.txt', '*.rtf', '*.odt', '*.pages',
                '*.xls', '*.xlsx', '*.csv', '*.ods', '*.numbers', '*.ppt', '*.pptx',
                '*.odp', '*.key', '*.epub', '*.mobi', '*.azw', '*.azw3',
            ],
            'archive': [
                '*.zip', '*.rar', '*.7z', '*.tar', '*.gz', '*.bz2', '*.xz',
                '*.tgz', '*.tbz2', '*.lz', '*.lzma', '*.z', '*.lzh',
            ],
        }
        
        # Camera brand patterns
        self.camera_patterns = {
            'canon': ['IMG_*.CR2', 'IMG_*.CR3', 'IMG_*.JPG'],
            'nikon': ['DSC_*.NEF', 'DSC_*.JPG'],
            'sony': ['DSC*.ARW', 'DSC*.JPG'],
            'fujifilm': ['DSCF*.RAF', 'DSCF*.JPG'],
            'olympus': ['P*.ORF', 'P*.JPG'],
            'panasonic': ['P*.RW2', 'P*.JPG'],
            'iphone': ['IMG_*.HEIC', 'IMG_*.MOV'],
            'samsung': ['IMG_*.JPG', 'VID_*.MP4'],
        }
        
        # Compiled patterns cache
        self.compiled_patterns = {}
        self._compile_all_patterns()
    
    def _compile_all_patterns(self):
        """Compile all patterns for fast matching"""
        for category, patterns in self.file_type_patterns.items():
            self.compiled_patterns[category] = []
            for pattern in patterns:
                try:
                    compiled = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
                    self.compiled_patterns[category].append(compiled)
                except Exception as e:
                    logger.error(f"Failed to compile pattern '{pattern}': {e}")
    
    def get_file_type(self, path: Path) -> Optional[str]:
        """
        Determine file type from pattern matching
        
        Args:
            path: Path to file
            
        Returns:
            File type string or None
        """
        filename = path.name.lower()
        
        for file_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.match(filename):
                    return file_type
        
        return None
    
    def get_camera_brand(self, path: Path) -> Optional[str]:
        """
        Guess camera brand from filename pattern
        
        Args:
            path: Path to file
            
        Returns:
            Camera brand string or None
        """
        filename = path.name.upper()
        
        for brand, patterns in self.camera_patterns.items():
            for pattern in patterns:
                if fnmatch.fnmatch(filename, pattern):
                    return brand
        
        return None
    
    def is_burst_photo(self, path: Path) -> bool:
        """
        Check if file appears to be part of a burst sequence
        
        Args:
            path: Path to file
            
        Returns:
            True if file looks like burst photo
        """
        filename = path.stem.upper()
        
        # Common burst patterns
        burst_patterns = [
            # iOS burst: IMG_1234, IMG_1234 2, IMG_1234 3
            r'^IMG_\d{4}(?: \d+)?$',
            
            # Samsung burst: IMG_20210101_123456_BURST001
            r'^IMG_\d{8}_\d{6}_BURST\d+$',
            
            # Android burst: 20210101_123456_001
            r'^\d{8}_\d{6}_\d{3}$',
            
            # Camera burst: DSC_1234, DSC_1234-2
            r'^DSC_\d+(?:-\d+)?$',
        ]
        
        for pattern in burst_patterns:
            if re.match(pattern, filename):
                return True
        
        return False
    
    def is_live_photo(self, path: Path) -> bool:
        """
        Check if file appears to be a Live Photo
        
        Args:
            path: Path to file
            
        Returns:
            True if file looks like Live Photo
        """
        # Live Photos are typically HEIC images with matching MOV file
        if path.suffix.lower() in ['.heic', '.jpg', '.jpeg']:
            mov_path = path.with_suffix('.MOV')
            if mov_path.exists():
                return True
        
        return False
    
    def find_related_files(self, path: Path) -> List[Path]:
        """
        Find files related to given file (burst, live photo, raw+jpg)
        
        Args:
            path: Path to file
            
        Returns:
            List of related file paths
        """
        related_files = []
        parent_dir = path.parent
        
        # Check for RAW+JPG pairs
        if path.suffix.lower() in ['.cr2', '.cr3', '.nef', '.arw', '.raf', '.orf', '.rw2']:
            # Look for matching JPG
            jpg_path = path.with_suffix('.JPG')
            if jpg_path.exists():
                related_files.append(jpg_path)
        
        elif path.suffix.lower() in ['.jpg', '.jpeg']:
            # Look for matching RAW
            for raw_ext in ['.CR2', '.CR3', '.NEF', '.ARW', '.RAF', '.ORF', '.RW2']:
                raw_path = path.with_suffix(raw_ext)
                if raw_path.exists():
                    related_files.append(raw_path)
        
        # Check for Live Photo video
        if path.suffix.lower() in ['.heic', '.jpg', '.jpeg']:
            mov_path = path.with_suffix('.MOV')
            if mov_path.exists():
                related_files.append(mov_path)
        
        # Check for burst sequences
        if self.is_burst_photo(path):
            # Look for files with same base pattern
            base_pattern = re.sub(r'(?: \d+$|-\d+$|_BURST\d+$)', '', path.stem)
            
            for sibling in parent_dir.iterdir():
                if sibling == path:
                    continue
                
                sibling_stem = sibling.stem
                sibling_base = re.sub(r'(?: \d+$|-\d+$|_BURST\d+$)', '', sibling_stem)
                
                if sibling_base == base_pattern and sibling.suffix.lower() == path.suffix.lower():
                    related_files.append(sibling)
        
        return related_files
    
    def match_pattern(self, path: Path, pattern: str) -> bool:
        """
        Check if path matches a specific pattern
        
        Args:
            path: Path to check
            pattern: Pattern string (supports wildcards)
            
        Returns:
            True if path matches pattern
        """
        try:
            return fnmatch.fnmatch(str(path), pattern)
        except Exception as e:
            logger.error(f"Error matching pattern '{pattern}': {e}")
            return False
    
    def get_all_patterns(self) -> Dict[str, List[str]]:
        """Get all configured patterns"""
        return {
            'file_types': self.file_type_patterns,
            'cameras': self.camera_patterns,
        }