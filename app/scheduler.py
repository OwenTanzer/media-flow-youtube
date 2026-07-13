"""Optional in-process scheduler: when ENABLE_SCHEDULER=true, runs the
Drive-queue batch job on a cron schedule inside the same web service, so
you don't need a second Railway service just for periodic processing."""

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .batch import run_batch
from .config import ConfigError, settings

logger = logging.getLogger("media_flow.scheduler")

_scheduler: BackgroundScheduler | None = None


def _run_scheduled_batch() -> None:
    try:
        results = run_batch()
        logger.info("Scheduled batch run processed %d video(s).", len(results))
    except ConfigError as exc:
        logger.error("Scheduled batch run skipped: %s", exc)
    except Exception:  # noqa: BLE001
        logger.exception("Scheduled batch run failed.")


def start_scheduler() -> None:
    global _scheduler
    if not settings.schedule_cron:
        logger.warning("ENABLE_SCHEDULER is true but SCHEDULE_CRON is not set; scheduler not started.")
        return
    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler(timezone="UTC")
    trigger = CronTrigger.from_crontab(settings.schedule_cron)
    _scheduler.add_job(_run_scheduled_batch, trigger=trigger, id="drive-queue-batch", replace_existing=True)
    _scheduler.start()
    logger.info("Scheduler started with cron schedule: %s", settings.schedule_cron)
