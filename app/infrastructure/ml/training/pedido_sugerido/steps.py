import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from app.domain.core.config import tz_now
from app.domain.models import VersionModelo
from app.domain.core.logging import logger
from app.domain.ml.abstractions.data_source_abc import DataSourceABC
from app.domain.ml.abstractions.step_abc import StepABC
from app.domain.ml.pipeline_context import TrainingContext
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.infrastructure.ml.models.pedido_sugerido_model import PedidoSugeridoModel
from app.domain.ml.training_params import SearchCVConfig
from app.infrastructure.ml.training.pedido_sugerido.constants import (
    CAT_FEATURES,
    HISTORIAL_VENTAS_COLS,
    RF_CANTIDAD_TARGET,
    RF_FEATURES,
    MODEL_PATH_BASE,
    SAMPLE_FRAC_PARAMS,
)
from app.infrastructure.ml.training.pedido_sugerido.utils import (
    calcular_mejores_params_rf,
    calcular_nro_clusters_kmeans,
    calcular_nro_vecinos_knn,
    calcular_params_apriori,
    extraer_producto,
    filtrar_canastas_por_soporte,
)


class LoadDataStep(StepABC[TrainingContext]):
    """
    Carga el dataset crudo usando el DataSource inyectado.
    Puede ser SqlServerDataSource, ExcelDataSource, etc.
    Resultado: ctx.raw_data con todas las filas y columnas originales.
    """

    def __init__(self, data_source: DataSourceABC) -> None:
        self._data_source = data_source

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        ctx.raw_data = self._data_source.load()
        return ctx


class EdaCleanDataStep(StepABC[TrainingContext]):
    """
    Normaliza los nombres de columnas al formato snake_case interno y elimina
    filas con valores nulos en campos críticos (cliente_id, producto_id,
    cantidad_vendida). Sin estos tres campos no se puede construir ninguna
    feature ni target.
    Resultado: ctx.clean_data listo para el resto del pipeline.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.raw_data is None:
            ctx.errors.append("EdaCleanDataStep: raw_data es None.")
            return ctx

        df = ctx.raw_data.copy()

        df = df.rename(
            columns={
                "FechaVenta": "fecha_venta",
                "ID_Ruta": "ruta_id",
                "ID_Producto": "producto_id",
                "ID_Zona": "zona_id",
                "ID_Cliente": "cliente_id",
                "Producto": "nombre_producto",
                "CantidadVendida": "cantidad_vendida",
                "LineaProducto": "linea_producto",
                "Marca": "marca",
                "ClasificacionCliente": "clasificacion_cliente",
                "Nombre_Ruta": "nombre_ruta",
                "Nombre_Zona": "nombre_zona",
                "Sucursal": "sucursal",
                "Vendedor": "vendedor",
            }
        )

        antes = len(df)
        df = df.dropna(
            subset=["cliente_id", "producto_id", "cantidad_vendida", "fecha_venta"]
        )
        eliminadas = antes - len(df)
        logger.info(
            "limpieza_completada", filas_eliminadas=eliminadas, filas_utiles=len(df)
        )
        if eliminadas > 0:
            logger.warning("filas_con_nulos_eliminadas", cantidad=eliminadas)

        ctx.clean_data = df
        return ctx


class CalculoAtributosDerivadosStep(StepABC[TrainingContext]):
    """
    Calcula features derivadas a nivel de transacción a partir del historial
    de ventas ordenado por cliente-producto-fecha:

    - dias_entre_compras: días desde la compra anterior del mismo producto
      (0 si es la primera compra).
    - promedio_historico: cantidad promedio comprada de ese producto por ese cliente.
    - promedio_ultimas_3: media móvil de las últimas 3 compras, captura la
      tendencia reciente del cliente con el producto.
    - dias_desde_ultima_compra: recencia respecto a la fecha máxima del dataset.
    - dia_semana / mes: señales de estacionalidad.

    Resultado: ctx.clean_data con las columnas anteriores añadidas.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("CalculoAtributosDerivadosStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()
        df["fecha_venta"] = pd.to_datetime(df["fecha_venta"])
        df = df.sort_values(["cliente_id", "producto_id", "fecha_venta"])

        grp = df.groupby(["cliente_id", "producto_id"])

        # Frecuencia entre compras
        df["fecha_anterior"] = grp["fecha_venta"].shift(1)
        df["dias_entre_compras"] = (
            df["fecha_venta"] - df["fecha_anterior"]
        ).dt.days.fillna(0)

        # Promedio histórico completo
        df["promedio_historico"] = grp["cantidad_vendida"].transform("mean")

        # Promedio de las últimas 3 compras (mejor predictor de tendencia)
        df["promedio_ultimas_3"] = grp["cantidad_vendida"].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

        # Recencia
        fecha_max = df["fecha_venta"].max()
        ultima_compra = grp["fecha_venta"].transform("max")
        df["dias_desde_ultima_compra"] = (fecha_max - ultima_compra).dt.days

        # Estacionalidad
        df["dia_semana"] = df["fecha_venta"].dt.dayofweek
        df["mes"] = df["fecha_venta"].dt.month

        grp_cliente = df.groupby("cliente_id")
        # productos distintos
        df["num_productos_distintos"] = grp_cliente["producto_id"].transform("nunique")
        # importe total cliente
        df["importe_total_cliente"] = grp_cliente["cantidad_vendida"].transform("sum")
        # frecuencia promedio entre dias de compras
        df["frecuencia_promedio_cliente"] = grp_cliente["dias_entre_compras"].transform(
            "mean"
        )
        # cantidad diferente de productos comprados
        df["cantidad_productos_comprados"] = grp_cliente["producto_id"].transform(
            "count"
        )

        ctx.clean_data = df
        logger.info("features_derivadas_calculadas", n_columnas=len(df.columns))
        return ctx


class ClusteringKMeansStep(StepABC[TrainingContext]):
    """
    Segmenta los clientes en grupos de comportamiento de compra usando KMeans.
    Cada cliente se resume en un vector con: cantidad promedio comprada,
    promedio histórico, frecuencia entre compras, variedad de productos y
    meses activo. Se escala con StandardScaler antes de clustering.
    Ejemplo:
        Cluster 1 → clientes grandes (alto volumen)
        Cluster 2 → clientes frecuentes
        Cluster 3 → clientes pequeños
        Cluster 4 → clientes irregulares

    El segmento resultante se añade como columna a ctx.clean_data para que
    los modelos RandomForest puedan usarlo como feature.

    Artefacto guardado en ctx.extra["model_km"]:
        - model: objeto KMeans entrenado
        - scaler: StandardScaler fitteado
        - features: lista de features usadas
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("ClusteringKMeansStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()
        feats_km = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "num_productos_distintos",
            "meses_activo",
        ]

        data_km = (
            df.groupby("cliente_id")
            .agg(
                cantidad_vendida=("cantidad_vendida", "mean"),
                promedio_historico=("promedio_historico", "mean"),
                dias_entre_compras=(
                    "dias_entre_compras",
                    lambda x: x[x > 0].mean() if (x > 0).any() else 0,
                ),
                num_productos_distintos=("producto_id", "nunique"),
                meses_activo=("mes", "nunique"),
            )
            .reset_index()
            .fillna(0)
        )

        scaler_km = StandardScaler()
        x_km = scaler_km.fit_transform(data_km[feats_km])
        muestra_km = (
            pd.DataFrame(x_km).sample(frac=SAMPLE_FRAC_PARAMS, random_state=42).values
        )
        nro_clusters = calcular_nro_clusters_kmeans(muestra_km)

        model_km = KMeans(
            n_clusters=nro_clusters, init="k-means++", random_state=42, n_init="auto"
        )
        data_km["segmento"] = model_km.fit_predict(x_km)

        seg_map = data_km.set_index("cliente_id")["segmento"]

        df["segmento"] = df["cliente_id"].map(seg_map).fillna(-1).astype(int)
        ctx.clean_data = df

        model_km = {
            "model": model_km,
            "scaler": scaler_km,
            "features": feats_km,
        }
        ctx.extra["model_km"] = model_km
        return ctx


class VecinosCercanosKnnStep(StepABC[TrainingContext]):
    """
    Construye el modelo de vecinos cercanos (KNN) para encontrar clientes
    similares en tiempo de predicción. Es independiente de KMeansStep.

    Cada cliente se representa con:
    - Features numéricas (cantidad, frecuencia, actividad): escaladas con
      StandardScaler.
    - Features categóricas (sucursal, zona, ruta, clasificación): codificadas
      con OneHotEncoder. Incluyen contexto geográfico y comercial del cliente.

    Ambos vectores se concatenan y se entrena un NearestNeighbors con
    distancia coseno, que mide similitud de dirección independientemente
    de la magnitud (útil cuando los volúmenes de compra varían mucho).

    Artefacto guardado en ctx.extra["model_knn"]:
        - model: objeto NearestNeighbors entrenado
        - customers: lista ordenada de cliente_id (índice ↔ posición en KNN)
        - scaler: StandardScaler para features numéricas
        - enc_cat: OneHotEncoder para features categóricas
        - feats_num: lista de features numéricas
        - cat_features: lista de features categóricas
        - data: DataFrame con el perfil agregado de cada cliente

    El historial de compras por vecino se calcula en predicción directamente
    desde historial_ventas, evitando almacenar una matriz densa clientes×productos.

    n_neighbors = número de vecinos que vas a usar para comparar
    Ejemplo:
        n_neighbors = 3 → miras 3 clientes parecidos
        n_neighbors = 5 → miras 5 clientes
        n_neighbors = 10 → miras 10 clientes
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("VecinosCercanosKnnStep: clean_data es None.")
            return ctx

        feats_num_knn = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "num_productos_distintos",
            "meses_activo",
        ]
        feats_cat_knn = [
            "sucursal",
            "zona_id",
            "ruta_id",
            "clasificacion_cliente",
            "segmento",
        ]

        data_knn = (
            ctx.clean_data.groupby("cliente_id")
            .agg(
                cantidad_vendida=("cantidad_vendida", "mean"),
                promedio_historico=("promedio_historico", "mean"),
                dias_entre_compras=(
                    "dias_entre_compras",
                    lambda x: x[x > 0].mean() if (x > 0).any() else 0,
                ),
                num_productos_distintos=("producto_id", "nunique"),
                meses_activo=("mes", "nunique"),
                clasificacion_cliente=("clasificacion_cliente", "first"),
                sucursal=("sucursal", "first"),
                ruta_id=("ruta_id", "first"),
                zona_id=("zona_id", "first"),
                segmento=("segmento", "first"),
            )
            .reset_index()
            .fillna(0)
        )
        all_customers = data_knn["cliente_id"].tolist()

        scaler_knn = StandardScaler()
        enc_cat_knn = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        x_num_knn = scaler_knn.fit_transform(data_knn[feats_num_knn])
        x_cat_knn = enc_cat_knn.fit_transform(data_knn[feats_cat_knn].astype(str))
        x_knn = np.hstack([x_num_knn, x_cat_knn])

        n_muestra = max(100, int(len(x_knn) * SAMPLE_FRAC_PARAMS))
        indices_muestra = np.random.default_rng(42).choice(
            len(x_knn), size=n_muestra, replace=False
        )
        muestra_knn = x_knn[indices_muestra]
        n_neighbors = calcular_nro_vecinos_knn(muestra_knn)
        model_knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        model_knn.fit(x_knn)

        model_knn = {
            "model": model_knn,
            "customers": all_customers,
            "scaler": scaler_knn,
            "enc_cat": enc_cat_knn,
            "feats_num": feats_num_knn,
            "cat_features": feats_cat_knn,
            "data": data_knn,
        }
        ctx.extra["model_knn"] = model_knn
        return ctx


class ConjuntoReglasAprioriStep(StepABC[TrainingContext]):
    """
    Genera reglas de asociación entre productos usando el algoritmo Apriori.
    Opera sobre el historial de transacciones agrupado por cliente: cada
    cliente representa una "canasta" con los productos que compró alguna vez.

    Parámetros:
        - min_support: fracción mínima de clientes que compran el conjunto
          (default 0.05 = al menos 5% de clientes)
        - min_confidence: confianza mínima de la regla A→B
          (default 0.3 = al menos 30% de quienes compran A también compran B)
        - min_lift: lift mínimo para filtrar reglas triviales
          (default 1.0 = la regla debe aportar más que el azar)

    Artefacto guardado en ctx.extra["model_apriori"]:
        - rules: DataFrame con columnas antecedents, consequents,
                 confidence, lift, support
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("ConjuntoReglasAprioriStep: clean_data es None.")
            return ctx

        df = ctx.clean_data

        # Una canasta por cliente: productos únicos que compró
        canastas = (
            df.groupby("cliente_id")["producto_id"]
            .apply(lambda x: x.unique().tolist())
            .tolist()
        )

        params = calcular_params_apriori(canastas)
        min_support = params["min_support"]
        min_confidence = params["min_confidence"]
        min_lift = params["min_lift"]

        canastas = filtrar_canastas_por_soporte(canastas, min_support)

        # Crear matriz binaria de clientes-productos para Apriori
        te = TransactionEncoder()
        te_array = te.fit_transform(canastas)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)  # type: ignore

        # max_len=2: solo computa pares de productos (itemsets de tamaño 1 y 2).
        # Las reglas que usamos son siempre A→B (1 producto → 1 producto),
        # por lo que itemsets de tamaño 3+ son innecesarios y causan explosión de memoria.
        frequent_itemsets = apriori(
            df_encoded,
            min_support=min_support,
            use_colnames=True,
            max_len=2,
        )

        if frequent_itemsets.empty:
            ctx.errors.append(
                "ConjuntoReglasAprioriStep: no se encontraron itemsets frecuentes. Bajá min_support."
            )
            return ctx

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
            num_itemsets=len(df_encoded),
        )
        rules = rules[rules["lift"] >= min_lift].copy()

        # Conserva solo reglas simples: 1 producto → 1 producto
        es_regla_simple = rules["antecedents"].apply(len) == 1
        rules = rules[es_regla_simple].copy()
        rules["antecedent"] = rules["antecedents"].apply(extraer_producto)
        rules["consequent"] = rules["consequents"].apply(extraer_producto)
        rules = rules[["antecedent", "consequent", "support", "confidence", "lift"]]
        rules = rules.sort_values("confidence", ascending=False).reset_index(drop=True)

        ctx.extra["model_apriori"] = {"rules": rules}
        logger.info("apriori_reglas_generadas", total=len(rules))
        return ctx


class PrepareDataArbolesStep(StepABC[TrainingContext]):
    """
    Construye el DataFrame de entrenamiento para el RandomForestRegressor,
    con una fila por par (cliente_id, producto_id).

    Las features se agregan desde el historial de transacciones:
    promedios, recencia, estacionalidad, segmento del cliente, etc.

    El único target es cantidad_vendida (mean): cuánto compra el cliente
    de ese producto en promedio por transacción. Usado por RandomForestRegressor.

    Resultado: ctx.extra["data_rf_df"] con features y el target.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("PrepareDataArbolesStep: clean_data es None.")
            return ctx
        # Una fila por cliente-producto (nivel de predicción)
        data_rf_df = (
            ctx.clean_data.groupby(["cliente_id", "producto_id"])
            .agg(
                nombre_producto=("nombre_producto", "first"),
                cantidad_vendida=("cantidad_vendida", "mean"),
                promedio_historico=("promedio_historico", "mean"),
                promedio_ultimas_3=("promedio_ultimas_3", "last"),
                dias_entre_compras=("dias_entre_compras", "mean"),
                dias_desde_ultima_compra=("dias_desde_ultima_compra", "mean"),
                dia_semana=("dia_semana", "last"),
                mes=("mes", "last"),
                marca=("marca", "first"),
                linea_producto=("linea_producto", "first"),
                clasificacion_cliente=("clasificacion_cliente", "first"),
                ruta_id=("ruta_id", "first"),
                zona_id=("zona_id", "first"),
                sucursal=("sucursal", "first"),
                segmento=("segmento", "first"),
                num_productos_distintos=("num_productos_distintos", "first"),
                importe_total_cliente=("importe_total_cliente", "first"),
                frecuencia_promedio_cliente=(
                    "frecuencia_promedio_cliente",
                    "first",
                ),
                cantidad_productos_comprados=(
                    "cantidad_productos_comprados",
                    "first",
                ),
            )
            .reset_index()
        )

        logger.info("datos_rf_preparados", pares_cliente_producto=len(data_rf_df))

        ctx.extra["data_rf_df"] = data_rf_df
        return ctx


class EnsembleArbolesRandomForestStep(StepABC[TrainingContext]):
    """
    Entrena un RandomForestRegressor para predecir la cantidad sugerida de un
    producto para un cliente (target: cantidad_vendida promedio por transacción).

    Las features categóricas se codifican con OrdinalEncoder antes del
    entrenamiento. Valores desconocidos se mapean a -1.

    Artefacto guardado en ctx.extra["model_rf_cantidad"]:
        - model: RandomForestRegressor entrenado
        - encoder: OrdinalEncoder fitteado sobre CAT_FEATURES
        - features: lista de features usadas (RF_FEATURES)
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        data_rf_df = ctx.extra.get("data_rf_df")
        if data_rf_df is None:
            ctx.errors.append("EnsembleArbolesRandomForestStep: data_rf_df es None.")
            return ctx

        X = data_rf_df[RF_FEATURES].copy()
        y = data_rf_df[RF_CANTIDAD_TARGET].fillna(0)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[CAT_FEATURES] = enc.fit_transform(X[CAT_FEATURES].fillna("DESCONOCIDO"))

        X_muestra = X.sample(frac=SAMPLE_FRAC_PARAMS, random_state=42)
        y_muestra = y.loc[X_muestra.index]
        config_rf = calcular_mejores_params_rf(X_muestra, y_muestra, SearchCVConfig())
        model = RandomForestRegressor(
            n_estimators=config_rf["n_estimators"],
            max_depth=config_rf["max_depth"],
            max_features=config_rf["max_features"],
            min_samples_split=config_rf["min_samples_split"],
            min_samples_leaf=config_rf["min_samples_leaf"],
            random_state=42,
            n_jobs=-1,
        )
        logger.info("rf_entrenando", muestras=len(X))
        model.fit(X, y)
        logger.info("rf_entrenado", muestras=len(X))

        ctx.extra["model_rf_cantidad"] = {
            "model": model,
            "encoder": enc,
            "features": RF_FEATURES,
        }
        return ctx


class SaveModelStep(StepABC[TrainingContext]):
    """
    Serializa todos los artefactos del pipeline en un único archivo .pkl
    usando joblib. El archivo contiene un dict con la clave "artefactos"
    que agrupa todos los modelos e infraestructura necesaria para predecir:

        - model_km: segmentación KMeans (no usada en predicción actualmente)
        - model_knn: vecinos cercanos con scaler, encoder y perfil de clientes
        - model_rf_cantidad: regresor de cantidad con encoder y features
        - historial_ventas: historial completo de transacciones, usado en
          predicción para construir features de productos candidatos
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("SaveModelStep: clean_data es None.")
            return ctx

        artefactos = {
            "model_km": ctx.extra["model_km"],
            "model_knn": ctx.extra["model_knn"],
            "model_apriori": ctx.extra["model_apriori"],
            "model_rf_cantidad": ctx.extra["model_rf_cantidad"],
            "historial_ventas": ctx.clean_data[HISTORIAL_VENTAS_COLS],
        }

        path_model = f"{MODEL_PATH_BASE}modelo_{ctx.model_name}_{ctx.version}.pkl"
        joblib.dump({"artefactos": artefactos}, str(path_model))

        ctx.extra["path_model"] = path_model
        logger.info(
            "modelo_guardado",
            path=str(path_model),
            modelo=ctx.model_name,
            version=ctx.version,
        )
        return ctx


class RegistryModelStep(StepABC[TrainingContext]):
    """
    Carga el modelo recién guardado desde disco y lo registra en el
    model_registry en memoria, dejándolo disponible para el pipeline
    de predicción sin necesidad de reiniciar el servidor.
    También persiste los metadatos del entrenamiento en VersionModelo.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        path_model = ctx.extra.get("path_model")
        if path_model is None:
            ctx.errors.append("RegistryModelStep: path_model no encontrado.")
            return ctx

        path_model = str(path_model)
        meta_data = ModelMetadata(
            name=ctx.model_name,
            version=ctx.version,
            path_model=path_model,
        )
        model = PedidoSugeridoModel(metadata=meta_data)
        model.load(path_model)

        model_registry.register(
            name=ctx.model_name,
            model=model,
        )

        self._guardar_version_modelo(ctx, path_model)

        return ctx

    def _guardar_version_modelo(self, ctx: TrainingContext, path_model: str) -> None:
        cantidad_clientes = 0
        cantidad_productos = 0
        if ctx.clean_data is not None:
            cantidad_clientes = ctx.clean_data["cliente_id"].nunique()
            cantidad_productos = ctx.clean_data["producto_id"].nunique()

        VersionModelo.objects.filter(nombre_modelo=ctx.model_name).update(activo=False)

        VersionModelo.objects.create(
            nombre_modelo=ctx.model_name,
            version=ctx.version,
            entrenado_en=tz_now(),
            ruta_pkl=path_model,
            tipo_fuente_datos="historial_ventas",
            cantidad_clientes=cantidad_clientes,
            cantidad_productos=cantidad_productos,
            hiperparametros=ctx.hyperparams,
            activo=True,
        )
