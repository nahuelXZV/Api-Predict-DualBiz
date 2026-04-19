import app.application.jobs.handlers  # noqa: F401 — activa el auto-registro de handlers

from app.domain.utils.enums import TipoJob
from app.application.jobs.job_registry import get_handler
from app.domain.models.tarea_programada import TareaProgramada


class JobRunner:
    def run(self, tipo_job: TipoJob, tarea_programada: TareaProgramada) -> None:
        handler = get_handler(tipo_job)
        handler(tarea_programada)


job_runner = JobRunner()
