import xgboost as xgb

from sklearn.preprocessing import OrdinalEncoder


from app.domain.core.logging import logger
from app.domain.ml.base_step import BaseStep
from app.domain.ml.base_context import TrainingContext
from app.infrastructure.ml.training.pedido_sugerido.utils import calcular_mejores_params_xgb

XGB_FEATURES = [
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

XGB_CANTIDAD_TARGET = "cantidad_vendida"

CAT_FEATURES = [
    "nombre_producto",
    "marca",
    "linea_producto",
    "clasificacion_cliente",
    "sucursal",
]


class EnsembleArbolesXGBoostStep(BaseStep[TrainingContext]):
    """
    Entrena un XGBRegressor para predecir la cantidad sugerida de un producto
    para un cliente (target: cantidad_vendida promedio por transacción).

    Las features categóricas se codifican con OrdinalEncoder antes del
    entrenamiento. Valores desconocidos se mapean a -1.

    Artefacto guardado en ctx.extra["model_xgb_cantidad"]:
        - model: XGBRegressor entrenado
        - encoder: OrdinalEncoder fitteado sobre CAT_FEATURES
        - features: lista de features usadas (XGB_FEATURES)
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        data_xgb_df = ctx.extra.get("data_xgb_df")
        if data_xgb_df is None:
            ctx.errors.append("EnsembleArbolesXGBoostStep: data_xgb_df es None.")
            return ctx

        X = data_xgb_df[XGB_FEATURES].copy()
        y = data_xgb_df[XGB_CANTIDAD_TARGET].fillna(0)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[CAT_FEATURES] = enc.fit_transform(X[CAT_FEATURES].fillna("DESCONOCIDO"))

        config_xgb = calcular_mejores_params_xgb(X, y)
        model = xgb.XGBRegressor(
            n_estimators=config_xgb["n_estimators"],
            learning_rate=config_xgb["learning_rate"],
            max_depth=config_xgb["max_depth"],
            subsample=config_xgb["subsample"],
            colsample_bytree=config_xgb["colsample_bytree"],
            min_child_weight=config_xgb["min_child_weight"],
            gamma=config_xgb["gamma"],
            random_state=42,
            verbosity=0,
        )
        logger.info("xgb_entrenando", muestras=len(X))
        model.fit(X, y)
        logger.info("xgb_entrenado", muestras=len(X))
        model_xgb_cantidad = {
            "model": model,
            "encoder": enc,
            "features": XGB_FEATURES,
        }
        ctx.extra["model_xgb_cantidad"] = model_xgb_cantidad
        return ctx
