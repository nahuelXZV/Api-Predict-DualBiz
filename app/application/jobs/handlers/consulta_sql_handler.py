from app.application.jobs.job_registry import register_job
from app.application.services.fuente_datos_service import fuente_datos_service
from app.domain.core.logging import logger
from app.domain.models.tarea_programada import TareaProgramada
from app.domain.utils.enums import TipoJob
from app.infrastructure.data_sources.data_source_factory import DataSourceFactory
from app.infrastructure.data_writers.data_writer_factory import DataWriterFactory


@register_job(TipoJob.CONSULTA_SQL)
def handle(tarea_programada: TareaProgramada, ejecucion_id: int) -> None:
    fuente_id = getattr(tarea_programada, "fuente_datos_id", None)
    if fuente_id is None:
        raise ValueError(
            "La tarea programada para consulta SQL debe tener un atributo 'fuente_datos_id'."
        )

    fuente_datos = fuente_datos_service.obtener_fuente_datos(fuente_id)
    if fuente_datos is None:
        raise ValueError(
            f"Fuente de datos con id {fuente_id} no encontrada en el sistema."
        )

    parameters_fuente = fuente_datos_service.obtener_parametros_fuente_datos(fuente_id)
    datasource = DataSourceFactory.build(fuente_datos.tipo, parameters_fuente)
    data = datasource.load()

    tipo_conexion = parameters_fuente.get("conexion_escritura", "sqlserver")
    data_writer = DataWriterFactory.build(tipo_conexion, parameters_fuente)
    cantidad_registros = data_writer.insert_many(data)
    
    logger.info(f"Cantidad de registros insertados: {cantidad_registros}")
    logger.info("consulta_sql_job_data_obtenida")
