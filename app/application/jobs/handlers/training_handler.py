from app.domain.core.config import tz_now
from app.domain.utils.enums import TipoJob
from app.domain.dtos.training_dto import TrainRequestDTO
from app.application.jobs.job_registry import register_job
from app.application.ml.model_manager import model_manager
from app.domain.models.tarea_programada import TareaProgramada


@register_job(TipoJob.TRAINING)
def handle(tarea_programada: TareaProgramada) -> None:
    params = tarea_programada.get_params()

    request = TrainRequestDTO(
        model_name=params["model_name"],
        version=tz_now().strftime("%Y.%m.%d"),
        parameters=params,
        tarea_programada_id=tarea_programada.id,
    )
    result = model_manager.train(request)
    if not result.success:
        raise RuntimeError(f"Entrenamiento fallido: {result.errors}")
