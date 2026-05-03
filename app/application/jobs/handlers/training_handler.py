from app.domain.utils.enums import TipoJob
from app.domain.dtos.training_dto import TrainRequestDTO
from app.application.jobs.job_registry import register_job
from app.application.ml.model_manager import model_manager
from app.domain.models.tarea_programada import TareaProgramada


@register_job(TipoJob.TRAINING)
def handle(tarea_programada: TareaProgramada, ejecucion_id: int) -> None:

    request = TrainRequestDTO(
        tarea_programada=tarea_programada,
        ejecucion_id=ejecucion_id,
    )
    result = model_manager.train(request)

    if not result.success:
        raise Exception(f"Error en entrenamiento: {result.errors}")
