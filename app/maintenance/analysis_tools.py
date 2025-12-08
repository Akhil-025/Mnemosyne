# app/maintenance/analysis_tools.py
import logging
from app.db.session import SessionLocal
from app.db.models import File, ProcessingQueue, AnalysisResult

logger = logging.getLogger(__name__)

def force_reanalyze_all():
    session = SessionLocal()
    try:
        files = session.query(File).all()
        count = 0

        # Optional: clear old analysis results
        session.query(AnalysisResult).delete()

        for f in files:
            # reset file status
            f.processed = False
            f.analyzed_at = None
            f.error = None

            # create a new analyze task
            pq = ProcessingQueue(
                file_id=f.id,
                task_type="analyze",
                priority=10,   # higher priority for reanalyze
                status="pending",
            )
            session.add(pq)
            count += 1

        session.commit()
        logger.info(f"[MAINT] Marked {count} files for re-analysis (queue entries created)")
    finally:
        session.close()

    """
    Usage:

    (venv) python main.py --reanalyze-all
    (venv) python main.py

    """