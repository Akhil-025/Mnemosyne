#main.py

"""
Mnemosyne - Digital Life Archival System
"""
import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.utils.windows_compat import is_windows, get_windows_special_folder
from app.utils.config import Config
from app.db.session import init_database
from app.core.intelligence_engine import IntelligenceEngine
from app.processing.ingestion import FileIngestor
from app.processing.analysis_worker import AnalysisWorker
from app.watchdog.monitor import FileMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mnemosyne_windows.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main entry point for Windows"""
    if not is_windows():
        logger.error("This script is designed for Windows only")
        sys.exit(1)
    
    print("=" * 60)
    print("Mnemosyne - Digital Life Archival System")
    print("=" * 60)
    
    # Load configuration
    config = Config()
    
    # Set Windows-specific paths
    appdata = get_windows_special_folder('local_appdata')
    if appdata:
        config.system.data_dir = appdata / "Mnemosyne"
        config.system.temp_dir = Path("C:/Temp/Mnemosyne")
    
    # Create directories
    directories = [
        config.system.data_dir,
        config.system.temp_dir,
        config.paths.vault,
        config.paths.cache,
        Path("./data/thumbnails"),
        Path("./models"),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Initialize database
    db_manager = init_database(config.database.url)
    
    # Initialize components
    intelligence = IntelligenceEngine(config)
    ingestor = FileIngestor(config, db_manager)
    monitor = FileMonitor(config, ingestor)
    worker = AnalysisWorker(config, db_manager, intelligence, poll_interval=2.0)
    
    # Start services
    print("\nStarting Mnemosyne services...")
    print(f"Data directory: {config.system.data_dir}")
    print(f"Watching: {config.paths.source}")
    print(f"Output: {config.paths.output}")
    print(f"Ollama: {config.ollama.base_url}")
    
    worker_task = None

    try:
        # 1) Process existing files (fills files + processing_queue)
        print("\nProcessing existing files...")
        await ingestor.process_directory(config.paths.source)

        # 2) Start watchdog (for new files)
        await monitor.start()

        # 3) Start analysis worker in background
        worker_task = asyncio.create_task(worker.start())

        # 4) Keep running
        print("\nMnemosyne is running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except asyncio.CancelledError:
        pass

    except KeyboardInterrupt:
        print("\nShutting down...")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    finally:
        # CLEAN shutdown
        if worker:
            await worker.stop()
        if worker_task:
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass

        if monitor:
            await monitor.stop()

if __name__ == "__main__":
    # Set Windows event loop policy
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())