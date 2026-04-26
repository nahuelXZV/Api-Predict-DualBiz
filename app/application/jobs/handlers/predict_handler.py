from app.domain.utils.enums import TipoJob
from app.application.jobs.job_registry import register_job
from app.domain.models.tarea_programada import TareaProgramada


@register_job(TipoJob.PREDICT)
def handle(tarea_programada: TareaProgramada) -> None:
    # model_name = tarea_programada.nombre_modelo
    # params = dict()
    # result = model_manager.predict(model_name=model_name, data=params)
    # if "error" in result:
    #     raise RuntimeError(f"Predicción fallida: {result['error']}")
    ...