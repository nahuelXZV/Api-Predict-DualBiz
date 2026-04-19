from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.domain.core.config import settings
from app.domain.core.logging import logger
from app.application.services.job_service import JobService, job_service
from app.infrastructure.db.repositories.tarea_programada_repository import (
    TareaProgramadaRepository,
)


class JobScheduler:
    def __init__(
        self, service: JobService, tarea_repo: TareaProgramadaRepository
    ) -> None:
        self._scheduler = BackgroundScheduler(timezone=settings.timezone)
        self._service = service
        self._tarea_repo = tarea_repo

    def start(self) -> None:
        tareas = self._tarea_repo.list_active()
        programadas = 0

        for tarea in tareas:
            if not tarea.cron_schedule:
                continue
            try:
                self._scheduler.add_job(
                    self._service.ejecutar,
                    trigger=CronTrigger.from_crontab(
                        tarea.cron_schedule, timezone=settings.timezone
                    ),
                    args=[tarea.id],
                    id=f"tarea_{tarea.id}",
                    replace_existing=True,
                    misfire_grace_time=60,
                )
                programadas += 1
                logger.info(
                    "job_programado", tarea=tarea.nombre, cron=tarea.cron_schedule
                )
            except Exception as e:
                logger.error("job_programado_error", tarea=tarea.nombre, error=str(e))

        self._scheduler.start()
        logger.info("job_scheduler_iniciado", tareas_programadas=programadas)

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("job_scheduler_detenido")


job_scheduler = JobScheduler(
    service=job_service, tarea_repo=TareaProgramadaRepository()
)
