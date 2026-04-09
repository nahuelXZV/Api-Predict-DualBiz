from app.domain.core.config import settings

MODEL_PATH_BASE = settings.path_models
RF_FEATURES = [
    "nombre_producto",  # categorica → OrdinalEncoder
    "marca",  # categorica
    "linea_producto",  # categorica
    "clasificacion_cliente",  # categorica
    "sucursal",  # categorica
    "ruta_id",  # numerica
    "zona_id",  # numerica
    "promedio_historico",  # cuánto compra en promedio
    "promedio_ultimas_3",  # tendencia reciente (más importante que el histórico)
    "dias_entre_compras",  # frecuencia de compra
    "dias_desde_ultima_compra",  # recencia
    "dia_semana",  # estacionalidad
    "mes",  # estacionalidad
    "segmento",  # cluster al que pertenece el cliente
    "num_productos_distintos",  # variedad de productos que compra el cliente
    "importe_total_cliente",  # volumen total comprado por el cliente
    "frecuencia_promedio_cliente",  # frecuencia promedio entre compras del cliente
    "cantidad_productos_comprados",  # cantidad total de productos comprados por el cliente
]

RF_CANTIDAD_TARGET = "cantidad_vendida"

CAT_FEATURES = [
    "nombre_producto",
    "marca",
    "linea_producto",
    "clasificacion_cliente",
    "sucursal",
]
