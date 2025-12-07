"""
Windows compatibility utilities for Mnemosyne
"""
import os
import sys
import ctypes
import platform
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Any, Tuple, Optional
import logging
import shutil

logger = logging.getLogger(__name__)


def is_windows() -> bool:
    """Check if running on Windows"""
    return sys.platform == 'win32'


def get_windows_special_folder(folder_name: str) -> Optional[Path]:
    """Get Windows special folder path"""
    if not is_windows():
        logger.warning("get_windows_special_folder called on non-Windows system")
        return None
    
    folder_guids = {
        'desktop': '{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}',
        'documents': '{FDD39AD0-238F-46AF-ADB4-6C85480369C7}',
        'pictures': '{33E28130-4E1E-4676-835A-98395C3BC3BB}',
        'music': '{4BD8D571-6D19-48D3-BE97-422220080E43}',
        'videos': '{18989B1D-99B5-455B-841C-AB7C74E4DDFC}',
        'downloads': '{374DE290-123F-4565-9164-39C4925E467B}',
        'appdata': '{3EB685DB-65F9-4CF6-A03A-E3EF65729F3D}',
        'local_appdata': '{F1B32785-6FBA-4FCF-9D55-7B8E7F157091}',
        'programdata': '{62AB5D82-FDC1-4DC3-A9DD-070D1D495D97}',
        'temp': '{A520A1A4-1780-4FF6-BD18-167343C5AF16}',
    }
    
    try:
        # Try using ctypes with SHGetKnownFolderPath
        from ctypes import wintypes
        from ctypes import windll, POINTER
        
        # Load shell32
        shell32 = windll.shell32
        
        # Define SHGetKnownFolderPath function
        SHGetKnownFolderPath = shell32.SHGetKnownFolderPath
        SHGetKnownFolderPath.argtypes = [
            POINTER(wintypes.GUID),
            wintypes.DWORD,
            wintypes.HANDLE,
            POINTER(wintypes.LPWSTR)
        ]
        
        # Get GUID for folder
        folder_guid = folder_guids.get(folder_name.lower())
        if not folder_guid:
            logger.error(f"Unknown Windows folder: {folder_name}")
            return None
        
        # Convert GUID string to GUID structure
        guid = wintypes.GUID()
        ctypes.windll.ole32.CLSIDFromString(folder_guid, ctypes.byref(guid))
        
        # Get folder path
        path_ptr = wintypes.LPWSTR()
        result = SHGetKnownFolderPath(
            ctypes.byref(guid),
            0,  # KF_FLAG_DEFAULT
            None,
            ctypes.byref(path_ptr)
        )
        
        if result == 0:  # S_OK
            path = path_ptr.value
            ctypes.windll.ole32.CoTaskMemFree(path_ptr)
            return Path(path)
        
        logger.warning(f"SHGetKnownFolderPath failed for {folder_name}, result: {result}")
        
    except (AttributeError, OSError, ImportError) as e:
        logger.debug(f"SHGetKnownFolderPath failed: {e}")
    
    # Fallback to environment variables
    env_map = {
        'desktop': 'USERPROFILE\\Desktop',
        'documents': 'USERPROFILE\\Documents',
        'pictures': 'USERPROFILE\\Pictures',
        'music': 'USERPROFILE\\Music',
        'videos': 'USERPROFILE\\Videos',
        'downloads': 'USERPROFILE\\Downloads',
        'appdata': 'APPDATA',
        'local_appdata': 'LOCALAPPDATA',
        'programdata': 'ProgramData',
        'temp': 'TEMP',
    }
    
    if folder_name.lower() in env_map:
        env_var = folder_name.upper() if folder_name.lower() in ['appdata', 'localappdata', 'temp'] else None
        if env_var and env_var in os.environ:
            path = os.environ[env_var]
        else:
            template = env_map[folder_name.lower()]
            # Replace USERPROFILE with actual user profile path
            if 'USERPROFILE' in template:
                userprofile = os.environ.get('USERPROFILE', os.path.expanduser('~'))
                path = template.replace('USERPROFILE', userprofile)
            else:
                path = os.path.expanduser(f'~/{folder_name}')
        
        if path:
            return Path(path)
    
    return None


def create_windows_junction(source: Path, target: Path) -> bool:
    """Create Windows junction point (similar to symlink)"""
    if not is_windows():
        logger.warning("create_windows_junction called on non-Windows system")
        return False
    
    try:
        import subprocess
        
        # Use mklink to create junction
        cmd = ['cmd', '/c', 'mklink', '/J', str(target), str(source)]
        
        # Run as administrator if needed (for system directories)
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            logger.debug(f"Created junction: {source} -> {target}")
            return True
        else:
            logger.error(f"Failed to create junction: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating Windows junction: {e}")
        return False


def create_windows_symlink(source: Path, target: Path, is_directory: bool = False) -> bool:
    """Create Windows symbolic link"""
    if not is_windows():
        logger.warning("create_windows_symlink called on non-Windows system")
        return False
    
    try:
        import subprocess
        
        # Use mklink to create symlink
        link_type = '/D' if is_directory else ''
        cmd = ['cmd', '/c', 'mklink', link_type, str(target), str(source)]
        
        # Remove empty strings from command list
        cmd = [part for part in cmd if part]
        
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0:
            logger.debug(f"Created symlink: {source} -> {target}")
            return True
        else:
            # Check if we need administrator privileges
            if "privilege" in result.stderr.lower():
                logger.warning(f"Administrator privileges required for symlink: {source} -> {target}")
            
            logger.error(f"Failed to create symlink: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"Error creating Windows symlink: {e}")
        return False


def get_windows_drives() -> List[str]:
    """Get list of available drives on Windows"""
    if not is_windows():
        return []
    
    try:
        drives = []
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if bitmask & 1:
                drive_path = f"{letter}:\\"
                
                # Check if drive is accessible
                try:
                    if os.path.exists(drive_path):
                        drives.append(drive_path)
                except (OSError, PermissionError):
                    pass
            
            bitmask >>= 1
        
        return drives
        
    except Exception as e:
        logger.error(f"Error getting Windows drives: {e}")
        return []


def is_hidden_windows(path: Path) -> bool:
    """Check if file/folder is hidden on Windows"""
    if not is_windows():
        return False
    
    try:
        # FILE_ATTRIBUTE_HIDDEN = 0x2
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        return attrs != -1 and bool(attrs & 2)
        
    except Exception:
        return False


def get_file_attributes_windows(path: Path) -> Dict[str, bool]:
    """Get Windows file attributes"""
    if not is_windows():
        return {}
    
    try:
        # Windows file attribute constants
        FILE_ATTRIBUTE_ARCHIVE = 0x20
        FILE_ATTRIBUTE_COMPRESSED = 0x800
        FILE_ATTRIBUTE_DEVICE = 0x40
        FILE_ATTRIBUTE_DIRECTORY = 0x10
        FILE_ATTRIBUTE_ENCRYPTED = 0x4000
        FILE_ATTRIBUTE_HIDDEN = 0x2
        FILE_ATTRIBUTE_INTEGRITY_STREAM = 0x8000
        FILE_ATTRIBUTE_NORMAL = 0x80
        FILE_ATTRIBUTE_NOT_CONTENT_INDEXED = 0x2000
        FILE_ATTRIBUTE_NO_SCRUB_DATA = 0x20000
        FILE_ATTRIBUTE_OFFLINE = 0x1000
        FILE_ATTRIBUTE_READONLY = 0x1
        FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
        FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
        FILE_ATTRIBUTE_REPARSE_POINT = 0x400
        FILE_ATTRIBUTE_SPARSE_FILE = 0x200
        FILE_ATTRIBUTE_SYSTEM = 0x4
        FILE_ATTRIBUTE_TEMPORARY = 0x100
        FILE_ATTRIBUTE_VIRTUAL = 0x10000
        
        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        
        if attrs == -1:  # INVALID_FILE_ATTRIBUTES
            return {}
        
        return {
            'archive': bool(attrs & FILE_ATTRIBUTE_ARCHIVE),
            'compressed': bool(attrs & FILE_ATTRIBUTE_COMPRESSED),
            'device': bool(attrs & FILE_ATTRIBUTE_DEVICE),
            'directory': bool(attrs & FILE_ATTRIBUTE_DIRECTORY),
            'encrypted': bool(attrs & FILE_ATTRIBUTE_ENCRYPTED),
            'hidden': bool(attrs & FILE_ATTRIBUTE_HIDDEN),
            'integrity_stream': bool(attrs & FILE_ATTRIBUTE_INTEGRITY_STREAM),
            'normal': bool(attrs & FILE_ATTRIBUTE_NORMAL),
            'not_content_indexed': bool(attrs & FILE_ATTRIBUTE_NOT_CONTENT_INDEXED),
            'no_scrub_data': bool(attrs & FILE_ATTRIBUTE_NO_SCRUB_DATA),
            'offline': bool(attrs & FILE_ATTRIBUTE_OFFLINE),
            'readonly': bool(attrs & FILE_ATTRIBUTE_READONLY),
            'recall_on_data_access': bool(attrs & FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS),
            'recall_on_open': bool(attrs & FILE_ATTRIBUTE_RECALL_ON_OPEN),
            'reparse_point': bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT),
            'sparse_file': bool(attrs & FILE_ATTRIBUTE_SPARSE_FILE),
            'system': bool(attrs & FILE_ATTRIBUTE_SYSTEM),
            'temporary': bool(attrs & FILE_ATTRIBUTE_TEMPORARY),
            'virtual': bool(attrs & FILE_ATTRIBUTE_VIRTUAL),
        }
        
    except Exception as e:
        logger.error(f"Error getting Windows file attributes: {e}")
        return {}


def normalize_windows_path(path: str) -> str:
    """Normalize Windows path for database storage"""
    if not is_windows():
        return path.replace('\\', '/')
    
    # Convert to forward slashes for consistency
    path = path.replace('\\', '/')
    
    # Remove drive letter if present
    if len(path) > 2 and path[1] == ':':
        path = path[2:]
    
    # Remove leading slash if present
    if path.startswith('/'):
        path = path[1:]
    
    return path


def denormalize_windows_path(path: str, drive: str = 'C') -> str:
    """Convert normalized path back to Windows path"""
    if not is_windows():
        return path
    
    # Add drive letter back
    if not (len(path) > 1 and path[1] == ':'):
        path = f"{drive}:{path}"
    
    # Convert to backslashes
    path = path.replace('/', '\\')
    
    return path


def get_windows_username() -> str:
    """Get current Windows username"""
    if is_windows():
        return os.environ.get('USERNAME', 'unknown')
    return os.environ.get('USER', 'unknown')


def get_windows_computer_name() -> str:
    """Get Windows computer name"""
    if not is_windows():
        return platform.node()
    
    try:
        name = ctypes.create_unicode_buffer(256)
        size = ctypes.wintypes.DWORD(256)
        
        if ctypes.windll.kernel32.GetComputerNameW(name, ctypes.byref(size)):
            return name.value
        else:
            return platform.node()
    except Exception:
        return platform.node()


def is_admin_windows() -> bool:
    """Check if running as administrator on Windows"""
    if not is_windows():
        return False
    
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def run_as_admin_windows(command: str, args: List[str] = None) -> bool:
    """Run command as administrator on Windows"""
    if not is_windows():
        logger.warning("run_as_admin_windows called on non-Windows system")
        return False
    
    try:
        import subprocess
        import sys
        
        # Convert command and args to string
        if args:
            full_command = [command] + args
        else:
            full_command = [command]
        
        # Use ShellExecute with runas verb
        from ctypes import wintypes
        from ctypes import windll
        
        SEE_MASK_NOCLOSEPROCESS = 0x00000040
        SW_SHOW = 5
        
        # Convert command to proper format
        cmd_str = subprocess.list2cmdline(full_command)
        
        # ShellExecuteW parameters
        hwnd = None
        operation = "runas"  # Run as administrator
        file = sys.executable if command == "python" else command
        parameters = subprocess.list2cmdline(args) if args else ""
        directory = None
        show_cmd = SW_SHOW
        
        result = windll.shell32.ShellExecuteW(
            hwnd,
            operation,
            file,
            parameters,
            directory,
            show_cmd
        )
        
        # ShellExecute returns value > 32 if successful
        return result > 32
        
    except Exception as e:
        logger.error(f"Error running as administrator: {e}")
        return False


def get_windows_version() -> Tuple[str, str, str]:
    """Get Windows version information"""
    if not is_windows():
        return ("", "", "")
    
    try:
        import winreg
        
        # Open registry key for Windows version
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, 
                           r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") as key:
            
            # Get product name
            product_name = winreg.QueryValueEx(key, "ProductName")[0]
            
            # Get release ID
            release_id = winreg.QueryValueEx(key, "ReleaseId")[0] if \
                        winreg.QueryValueEx(key, "ReleaseId")[0] else \
                        winreg.QueryValueEx(key, "CurrentBuild")[0]
            
            # Get build number
            build_number = winreg.QueryValueEx(key, "CurrentBuildNumber")[0]
            
            return (product_name, release_id, build_number)
            
    except Exception as e:
        logger.error(f"Error getting Windows version: {e}")
        return ("Windows", "Unknown", "Unknown")


def fix_windows_long_paths() -> bool:
    """Enable Windows long path support if available"""
    if not is_windows():
        return False
    
    try:
        import winreg
        
        # Windows 10 Anniversary Update (1607) and later support long paths
        # Need to enable in registry: HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled
        
        key_path = r"SYSTEM\CurrentControlSet\Control\FileSystem"
        value_name = "LongPathsEnabled"
        
        try:
            # Try to open the key
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, 
                                winreg.KEY_READ | winreg.KEY_WRITE)
            
            # Get current value
            try:
                current_value, _ = winreg.QueryValueEx(key, value_name)
            except FileNotFoundError:
                current_value = 0
            
            # Enable if not already enabled
            if current_value != 1:
                winreg.SetValueEx(key, value_name, 0, winreg.REG_DWORD, 1)
                logger.info("Windows long path support enabled")
                return True
            else:
                logger.debug("Windows long path support already enabled")
                return True
                
        except PermissionError:
            logger.warning("Administrator privileges required to enable Windows long paths")
            return False
        except Exception as e:
            logger.error(f"Error enabling Windows long paths: {e}")
            return False
            
    except ImportError:
        logger.warning("winreg not available, cannot enable Windows long paths")
        return False


def create_windows_startup_shortcut(app_name: str, target_path: Path, 
                                   arguments: str = "") -> bool:
    """Create Windows startup shortcut"""
    if not is_windows():
        return False
    
    try:
        import pythoncom
        from win32com.client import Dispatch
        
        # Get startup folder
        startup_folder = get_windows_special_folder('startup')
        if not startup_folder:
            startup_folder = Path(os.environ.get('APPDATA', '')) / \
                           "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        
        # Create shortcut
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortcut(str(startup_folder / f"{app_name}.lnk"))
        
        shortcut.TargetPath = str(target_path)
        shortcut.Arguments = arguments
        shortcut.WorkingDirectory = str(target_path.parent)
        shortcut.Description = f"{app_name} Startup"
        shortcut.Save()
        
        logger.info(f"Created startup shortcut for {app_name}")
        return True
        
    except ImportError:
        logger.warning("pywin32 not installed, cannot create Windows shortcut")
        return False
    except Exception as e:
        logger.error(f"Error creating Windows startup shortcut: {e}")
        return False


# Windows-specific file operations
def safe_move_windows(source: Path, target: Path, use_hardlink: bool = True) -> bool:
    """
    Safe file move operation for Windows with hardlink fallback
    
    Args:
        source: Source file path
        target: Target file path
        use_hardlink: Attempt to create hardlink first
        
    Returns:
        True if successful, False otherwise
    """
    if not is_windows():
        # Use standard move on non-Windows
        try:
            shutil.move(str(source), str(target))
            return True
        except Exception as e:
            logger.error(f"Error moving file: {e}")
            return False
    
    try:
        # Check if same drive
        source_drive = source.drive
        target_drive = target.drive
        
        if use_hardlink and source_drive == target_drive:
            try:
                # Try to create hardlink
                os.link(source, target)
                # Remove source if hardlink successful
                source.unlink()
                return True
            except OSError:
                # Hardlink failed, fallback to move
                pass
        
        # Regular move
        shutil.move(str(source), str(target))
        return True
        
    except Exception as e:
        logger.error(f"Error moving file on Windows: {e}")
        return False