from app.domain.core.logging import logger
from app.domain.utils.enums import TipoJob, DisparadoPor
from app.application.jobs.job_runner import JobRunner, job_runner
from app.infrastructure.db.repositories.tarea_programada_repository import (
    TareaProgramadaRepository,
)
from app.infrastructure.db.repositories.ejecucion_tarea_repository import (
    EjecucionTareaRepository,
)


class JobService:
    def __init__(
        self,
        tarea_repo: TareaProgramadaRepository,
        ejecucion_repo: EjecucionTareaRepository,
        runner: JobRunner,
    ) -> None:
        self._tarea_repo = tarea_repo
        self._ejecucion_repo = ejecucion_repo
        self._runner = runner

    def ejecutar(
        self,
        tarea_id: int,
        disparado_por: DisparadoPor = DisparadoPor.SCHEDULER,
    ) -> None:
        tarea = self._tarea_repo.get_by_id(tarea_id)
        if tarea is None:
            raise ValueError(f"TareaProgramada {tarea_id} no encontrada.")

        logger.info(
            "job_iniciado",
            tarea=tarea.nombre,
            tipo=tarea.tipo_job,
            disparado_por=disparado_por.value,
        )

        ejecucion = self._ejecucion_repo.create_inicio(tarea_id, disparado_por.value)

        try:
            self._runner.run(TipoJob(tarea.tipo_job), tarea)
            self._ejecucion_repo.marcar_exitosa(ejecucion.id)
            logger.info("job_exitoso", tarea=tarea.nombre, ejecucion_id=ejecucion.id)
        except Exception as e:
            self._ejecucion_repo.marcar_fallida(ejecucion.id, str(e))
            logger.error(
                "job_fallido",
                tarea=tarea.nombre,
                ejecucion_id=ejecucion.id,
                error=str(e),
            )
            raise


job_service = JobService(
    tarea_repo=TareaProgramadaRepository(),
    ejecucion_repo=EjecucionTareaRepository(),
    runner=job_runner,
)
