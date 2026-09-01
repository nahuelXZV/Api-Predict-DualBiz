from app.application.jobs.job_registry import register_job
from app.application.ml.model_manager import model_manager
from app.application.services.clientes_service import cliente_service
from app.application.services.lote_prediccion_service import lote_prediccion_service
from app.application.services.predict_service import predict_service
from app.application.services.resultado_prediccion_service import (
    resultado_prediccion_service,
)
from app.application.services.version_modelo_service import version_modelo_service
from app.application.utils.parser import parse_bool, parse_int
from app.domain.core.logging import logger
from app.domain.dtos.predict_dto import PredictResponseDTO
from app.domain.models.tarea_programada import TareaProgramada
from app.domain.utils.enums import TipoJob


@register_job(TipoJob.PREDICT)
def handle(tarea_programada: TareaProgramada, ejecucion_id: int) -> None:

    parameters = tarea_programada.get_params()
    nombre_modelo = parameters["model_name"]
    modelo_activo = version_modelo_service.get_version_activa(nombre_modelo)

    if modelo_activo is None:
        raise RuntimeError(f"No se encontró modelo activo para {nombre_modelo}")

    lote_id = None
    cantidad_predicciones = 0

    # PREDICCIÓN
    try:
        lote_id = lote_prediccion_service.iniciar_lote(
            nombre_modelo=nombre_modelo,
            parametros=parameters,
        )

        clientes = cliente_service.listar_clientes()
        if not clientes:
            raise RuntimeError("No existen clientes para generar predicciones")

        base_parameters = {
            "cantidad_minima": parse_int(parameters.get("cantidad_minima"), 1),
            "top_n": parse_int(parameters.get("top_n"), 50),
            "porcentaje_pareto": parse_int(parameters.get("porcentaje_pareto"), 20),
            "solo_nuevos": parse_bool(parameters.get("solo_nuevos"), False),
            "recomendacion_apriori": parse_bool(
                parameters.get("recomendacion_apriori"), False
            ),
            "recomendacion_destacados": parse_bool(
                parameters.get("recomendacion_destacados"), False
            ),
        }

        for cliente in clientes:
            try:
                parameters = {
                    **base_parameters,
                    "cliente_id": parse_int(cliente.codigo_erp),
                }

                pred_result: PredictResponseDTO = predict_service.predict(
                    model_name=modelo_activo.nombre_modelo,
                    hyperparams=parameters,
                )

                if not pred_result or not pred_result.predictions:
                    continue

                resultado_prediccion_service.guardar_resultados(
                    lote_id=lote_id,
                    resultados=pred_result.predictions,
                )

                cantidad_predicciones += len(pred_result.predictions)

            except Exception as e:
                logger.error(
                    "Error prediciendo cliente %s: %s",
                    cliente.codigo_erp,
                    e,
                    exc_info=True,
                )
                continue

        lote_prediccion_service.completar_lote(
            lote_id,
            cantidad_predicciones,
        )

    except Exception:
        if lote_id is not None:
            lote_prediccion_service.fallar_lote(lote_id)
        logger.exception("Falló el job PREDICT")
        raise
