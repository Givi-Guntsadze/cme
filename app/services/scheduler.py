"""
Scheduler service using APScheduler.

Runs background jobs for refreshing CME sources periodically.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable

from sqlmodel import Session, select

logger = logging.getLogger(__name__)

# We'll use a simple asyncio-based scheduler
# APScheduler would be the full solution, but avoiding the dependency for now
_scheduled_jobs: dict[str, dict] = {}
_scheduler_running = False
_scheduler_task: asyncio.Task | None = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def refresh_sources():
    """
    Weekly job to re-crawl all monitored sources.

    Process:
    1. Fetch all ScrapeSource URLs
    2. Trigger Apify run
    3. Update DB with new/changed activities
    4. Archive expired activities
    """
    from ..db import get_session
    from ..models import Activity
    from .scraper import crawl_and_extract_activities

    logger.info("Starting scheduled source refresh...")

    try:
        # Import ScrapeSource here to avoid circular imports
        # This model will be added to models.py
        try:
            from ..models import ScrapeSource
        except ImportError:
            logger.warning("ScrapeSource model not yet available")
            return

        with get_session() as session:
            sources = list(session.exec(
                select(ScrapeSource).where(ScrapeSource.enabled == True)
            ))

            if not sources:
                logger.info("No enabled scrape sources found")
                return

            urls = [s.url for s in sources if s.url]
            logger.info("Refreshing %d sources: %s", len(urls), urls)

            # Crawl all sources
            new_activities = await crawl_and_extract_activities(urls, max_pages=100)

            # Process results - basic deduplication by URL
            existing_urls = set()
            for act in session.exec(select(Activity.url).where(Activity.url.isnot(None))):
                if act:
                    existing_urls.add(act)

            inserted = 0
            for activity in new_activities:
                if activity.url and activity.url not in existing_urls:
                    session.add(activity)
                    existing_urls.add(activity.url)
                    inserted += 1

            # Update last_crawled timestamp for sources
            for source in sources:
                source.last_crawled = utcnow()
                session.add(source)

            session.commit()
            logger.info("Refresh complete: inserted %d new activities", inserted)

    except Exception:
        logger.exception("Source refresh failed")


async def _scheduler_loop():
    """Simple scheduler loop that runs jobs at configured intervals."""
    global _scheduler_running

    while _scheduler_running:
        try:
            now = utcnow()

            for job_id, job_info in list(_scheduled_jobs.items()):
                next_run = job_info.get("next_run")
                if next_run and now >= next_run:
                    func = job_info.get("func")
                    interval = job_info.get("interval_seconds", 604800)  # Default: 1 week

                    if func:
                        logger.info("Running scheduled job: %s", job_id)
                        try:
                            if asyncio.iscoroutinefunction(func):
                                await func()
                            else:
                                func()
                        except Exception:
                            logger.exception("Job %s failed", job_id)

                    # Schedule next run
                    from datetime import timedelta
                    job_info["next_run"] = now + timedelta(seconds=interval)

        except Exception:
            logger.exception("Scheduler loop error")

        # Check every minute
        await asyncio.sleep(60)


def schedule_job(
    job_id: str,
    func: Callable,
    interval_seconds: int = 604800,  # Default: weekly
    run_immediately: bool = False,
):
    """
    Schedule a recurring job.

    Args:
        job_id: Unique identifier for the job
        func: Function to call (can be async)
        interval_seconds: Seconds between runs (default: 1 week)
        run_immediately: If True, run the job immediately on first start
    """
    from datetime import timedelta

    next_run = utcnow()
    if not run_immediately:
        next_run = next_run + timedelta(seconds=interval_seconds)

    _scheduled_jobs[job_id] = {
        "func": func,
        "interval_seconds": interval_seconds,
        "next_run": next_run,
    }
    logger.info("Scheduled job '%s' with interval %ds", job_id, interval_seconds)


def remove_job(job_id: str):
    """Remove a scheduled job."""
    if job_id in _scheduled_jobs:
        del _scheduled_jobs[job_id]
        logger.info("Removed job '%s'", job_id)


async def start_scheduler():
    """Start the background scheduler."""
    global _scheduler_running, _scheduler_task

    if _scheduler_running:
        logger.warning("Scheduler already running")
        return

    _scheduler_running = True

    # Schedule the default refresh job (weekly on Mondays at 3 AM)
    schedule_job(
        "refresh_sources",
        refresh_sources,
        interval_seconds=604800,  # 1 week
        run_immediately=False,
    )

    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler started")


async def stop_scheduler():
    """Stop the background scheduler."""
    global _scheduler_running, _scheduler_task

    _scheduler_running = False

    if _scheduler_task:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None

    logger.info("Scheduler stopped")


def init_scheduler():
    """Initialize scheduler - call from lifespan context."""
    # This is a sync wrapper for compatibility
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(start_scheduler())
        else:
            loop.run_until_complete(start_scheduler())
    except RuntimeError:
        # No event loop
        asyncio.run(start_scheduler())
