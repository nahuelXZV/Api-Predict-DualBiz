import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from app.domain.core.config import settings, tz_now
from app.domain.core.logging import logger
from app.domain.models import TareaProgramada
from app.domain.utils.enums import DisparadoPor
from app.application.services.job_service import JobService, job_service
from app.infrastructure.db.repositories.tarea_programada_repository import (
    TareaProgramadaRepository,
)

REINTENTO_DELAY_SEGUNDOS = 300


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
        self._service.ejecutar(tarea_id, disparado_por=DisparadoPor.MANUAL)

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
            self._ejecutar_con_reintentos,
            trigger=CronTrigger.from_crontab(
                tarea.cron_schedule, timezone=settings.timezone
            ),
            args=[tarea.id],
            id=f"tarea_{tarea.id}",
            replace_existing=True,
            misfire_grace_time=60,
        )
        logger.info("job_programado", tarea=tarea.nombre, cron=tarea.cron_schedule)

    def _ejecutar_con_reintentos(
        self,
        tarea_id: int,
        numero_intento: int = 1,
        ejecucion_original_id: int | None = None,
    ) -> None:
        tarea = self._tarea_repo.get_by_id(tarea_id)
        if tarea is None:
            logger.error("reintento_tarea_no_encontrada", tarea_id=tarea_id)
            return

        disparado_por = (
            DisparadoPor.REINTENTO if numero_intento > 1 else DisparadoPor.SCHEDULER
        )

        try:
            self._service.ejecutar(
                tarea_id,
                disparado_por=disparado_por,
                numero_intento=numero_intento,
                ejecucion_original_id=ejecucion_original_id,
            )
        except Exception:
            if numero_intento < tarea.max_reintentos + 1:
                proximo_intento = numero_intento + 1
                run_at = tz_now() + datetime.timedelta(seconds=REINTENTO_DELAY_SEGUNDOS)
                job_id = f"reintento_{tarea_id}_{proximo_intento}"

                # ejecucion_original_id apunta siempre al primer intento
                original_id = ejecucion_original_id

                self._scheduler.add_job(
                    self._ejecutar_con_reintentos,
                    trigger=DateTrigger(run_date=run_at),
                    args=[tarea_id, proximo_intento, original_id],
                    id=job_id,
                    replace_existing=True,
                )
                logger.info(
                    "reintento_programado",
                    tarea=tarea.nombre,
                    intento=proximo_intento,
                    max=tarea.max_reintentos,
                    run_at=run_at.isoformat(),
                )


job_scheduler = JobScheduler(
    service=job_service, tarea_repo=TareaProgramadaRepository()
)
