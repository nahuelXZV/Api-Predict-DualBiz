from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.domain.core.config import settings
from app.domain.core.logging import logger
from app.domain.models import TareaProgramada
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
                self._agregar(tarea)
                programadas += 1
            except Exception as e:
                logger.error("job_programado_error", tarea=tarea.nombre, error=str(e))

        self._scheduler.start()
        logger.info("job_scheduler_iniciado", tareas_programadas=programadas)

    def agregar_o_actualizar(self, tarea_id: int) -> None:
        tarea = self._tarea_repo.get_by_id(tarea_id)
        if tarea is None or not tarea.activo or not tarea.cron_schedule:
            self.eliminar(tarea_id)
            return
        try:
            self._agregar(tarea)
            logger.info("job_actualizado", tarea=tarea.nombre, cron=tarea.cron_schedule)
        except Exception as e:
            logger.error("job_actualizado_error", tarea_id=tarea_id, error=str(e))

    def eliminar(self, tarea_id: int) -> None:
        job_id = f"tarea_{tarea_id}"
        if self._scheduler.get_job(job_id):
            self._scheduler.remove_job(job_id)
            logger.info("job_eliminado", tarea_id=tarea_id)

    def ejecutar_ahora(self, tarea_id: int) -> None:
        self._service.ejecutar(tarea_id)

    def listar(self) -> list[dict]:
        return [
            {
                "id": job.id,
                "next_run": job.next_run_time,
                "trigger": str(job.trigger),
            }
            for job in self._scheduler.get_jobs()
        ]

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("job_scheduler_detenido")

    def _agregar(self, tarea: TareaProgramada) -> None:
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
        logger.info("job_programado", tarea=tarea.nombre, cron=tarea.cron_schedule)


job_scheduler = JobScheduler(
    service=job_service, tarea_repo=TareaProgramadaRepository()
)
