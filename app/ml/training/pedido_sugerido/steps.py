import joblib
import numpy as np
import pandas as pd

import xgboost as xgb
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from app.domain.ml.base_step import BaseStep
from app.domain.ml.base_context import TrainingContext
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.ml.models.pedido_sugerido_model import PedidoSugeridoModel

BASE_DIR = Path(__file__).resolve().parents[4]
DATA_PATH = BASE_DIR / "storage" / "data" / "consulta_base.csv"
MODEL_PATH_BASE = BASE_DIR / "storage" / "models"

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
]

XGB_CANTIDAD_TARGET = "cantidad_vendida"

CAT_FEATURES = [
    "nombre_producto",
    "marca",
    "linea_producto",
    "clasificacion_cliente",
    "sucursal",
]


class LoadDataStep(BaseStep[TrainingContext]):
    """
    Carga el dataset crudo desde el CSV de ventas.
    Resultado: ctx.raw_data con todas las filas y columnas originales.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset original: {len(df):,} filas | {df.shape[1]} columnas")
        ctx.raw_data = df
        return ctx


class EDA_CleanDataStep(BaseStep[TrainingContext]):
    """
    Normaliza los nombres de columnas al formato snake_case interno y elimina
    filas con valores nulos en campos críticos (cliente_id, nombre_producto,
    cantidad_vendida). Sin estos tres campos no se puede construir ninguna
    feature ni target.
    Resultado: ctx.clean_data listo para el resto del pipeline.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.raw_data is None:
            ctx.errors.append("CleanDataStep: raw_data es None.")
            return ctx

        df = ctx.raw_data.copy()

        df = df.rename(
            columns={
                "FechaVenta": "fecha_venta",
                "ID_Ruta": "ruta_id",
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
        df = df.dropna(subset=["cliente_id", "nombre_producto", "cantidad_vendida"])
        print(f"Limpieza: {antes - len(df):,} filas eliminadas → {len(df):,} útiles")

        ctx.clean_data = df
        return ctx


class CalculoAtributosDerivadosStep(BaseStep[TrainingContext]):
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
            ctx.errors.append("AddDerivedFeatureStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()
        df["fecha_venta"] = pd.to_datetime(df["fecha_venta"])
        df = df.sort_values(["cliente_id", "nombre_producto", "fecha_venta"])

        grp = df.groupby(["cliente_id", "nombre_producto"])

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

        ctx.clean_data = df
        print(f"Features derivadas calculadas. Columnas: {list(df.columns)}")
        return ctx


def CalcularNroClusterKMeans(data: pd.DataFrame) -> int:
    inercias = []
    for k in range(2, 8 + 1):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(data)
        inercias.append(km.inertia_)

    segunda_derivada = np.diff(np.diff(inercias))
    nro_clusters_calculados = int(np.argmax(segunda_derivada)) + 4
    print(f"Total de clusters calculados: {nro_clusters_calculados}")
    return nro_clusters_calculados


class Clustering_KMeansStep(BaseStep[TrainingContext]):
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
    los modelos XGBoost puedan usarlo como feature.

    Artefacto guardado en ctx.extra["model_km"]:
        - model: objeto KMeans entrenado
        - scaler: StandardScaler fitteado
        - features: lista de features usadas
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("KMeansStep: clean_data es None.")
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
                dias_entre_compras=("dias_entre_compras", "mean"),
                num_productos_distintos=("nombre_producto", "nunique"),
                meses_activo=("mes", "nunique"),
            )
            .reset_index()
            .fillna(0)
        )

        scaler_km = StandardScaler()
        x_km = scaler_km.fit_transform(data_km[feats_km])

        nro_clusters = CalcularNroClusterKMeans(x_km)

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


def CalcularNroVecinosKnn(data: pd.DataFrame, k_min: int = 3, k_max: int = 51) -> int:
    """
    Usa la curva de distancias promedio para encontrar el número óptimo
    de vecinos. Similar al método del codo en KMeans.
    """
    k_max = min(k_max, len(data) - 1)
    k_range = range(k_min, k_max + 1, 2)  # solo impares: 3, 5, 7, ...
    distancias_promedio = []

    for k in k_range:
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(data)
        distancias, _ = nn.kneighbors(data)
        distancias_promedio.append(
            distancias[:, -1].mean()
        )  # dist al vecino más lejano

    segunda_derivada = np.diff(np.diff(distancias_promedio))
    k_optimo = list(k_range)[int(np.argmax(segunda_derivada)) + 2]

    print(f"KNN: k óptimo por codo = {k_optimo}")
    return k_optimo


class VecinosCercanos_KnnStep(BaseStep[TrainingContext]):
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

    n_neighbors = número de vecinos que vas a usar para comparar
    Ejemplo:
        n_neighbors = 3 → miras 3 clientes parecidos
        n_neighbors = 5 → miras 5 clientes
        n_neighbors = 10 → miras 10 clientes
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("KnnStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()
        feats_num_knn = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "meses_activo",
        ]
        feats_cat_knn = ["sucursal", "zona_id", "ruta_id", "clasificacion_cliente"]

        data_knn = (
            df.groupby("cliente_id")
            .agg(
                cantidad_vendida=("cantidad_vendida", "mean"),
                promedio_historico=("promedio_historico", "mean"),
                dias_entre_compras=("dias_entre_compras", "mean"),
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

        n_neighbors = CalcularNroVecinosKnn(x_knn)
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


class ConjuntoReglasApriori_Step(BaseStep[TrainingContext]):
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
            ctx.errors.append("AprioriStep: clean_data es None.")
            return ctx

        min_support = ctx.extra.get("apriori_min_support", 0.05)
        min_confidence = ctx.extra.get("apriori_min_confidence", 0.3)
        min_lift = ctx.extra.get("apriori_min_lift", 1.0)

        df = ctx.clean_data

        # Una canasta por cliente: lista de productos que compró
        canastas = df.groupby("cliente_id")["nombre_producto"].apply(list).tolist()

        te = TransactionEncoder()
        te_array = te.fit_transform(canastas)
        df_encoded = pd.DataFrame(te_array, columns=te.columns_)

        frequent_itemsets = apriori(
            df_encoded,
            min_support=min_support,
            use_colnames=True,
        )

        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=min_confidence,
        )
        rules = rules[rules["lift"] >= min_lift].copy()

        # Simplifica: antecedente y consecuente como strings (1 producto c/u)
        rules = rules[rules["antecedents"].apply(len) == 1].copy()
        rules["antecedent"] = rules["antecedents"].apply(lambda x: list(x)[0])
        rules["consequent"] = rules["consequents"].apply(lambda x: list(x)[0])
        rules = rules[["antecedent", "consequent", "support", "confidence", "lift"]]
        rules = rules.sort_values("confidence", ascending=False).reset_index(drop=True)

        ctx.extra["model_apriori"] = {"rules": rules}
        print(f"AprioriStep: {len(rules)} reglas generadas")
        return ctx


class PrepareDataXGBStep(BaseStep[TrainingContext]):
    """
    Construye el DataFrame de entrenamiento para los modelos XGBoost,
    con una fila por par (cliente_id, nombre_producto).

    Las features se agregan desde el historial de transacciones:
    promedios, recencia, estacionalidad, segmento del cliente, etc.

    El único target es cantidad_vendida (mean): cuánto compra el cliente
    de ese producto en promedio por transacción. Usado por XGBRegressor.

    Resultado: ctx.extra["data_xgb_df"] con features y el target.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("PrepareDataXGBoostStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()

        # Una fila por cliente-producto (nivel de predicción)
        data_xgb_df = (
            df.groupby(["cliente_id", "nombre_producto"])
            .agg(
                cantidad_vendida=("cantidad_vendida", "mean"),
                promedio_historico=("promedio_historico", "mean"),
                promedio_ultimas_3=("promedio_ultimas_3", "last"),
                dias_entre_compras=("dias_entre_compras", "mean"),
                dias_desde_ultima_compra=("dias_desde_ultima_compra", "mean"),
                dia_semana=("dia_semana", "first"),
                mes=("mes", "first"),
                marca=("marca", "first"),
                linea_producto=("linea_producto", "first"),
                clasificacion_cliente=("clasificacion_cliente", "first"),
                ruta_id=("ruta_id", "first"),
                zona_id=("zona_id", "first"),
                sucursal=("sucursal", "first"),
                segmento=("segmento", "first"),
            )
            .reset_index()
        )

        print(f"PrepareDataXGBStep: {len(data_xgb_df):,} pares cliente-producto")

        ctx.extra["data_xgb_df"] = data_xgb_df
        return ctx


class EnsembleArboles_XGBoostStep(BaseStep[TrainingContext]):
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
            ctx.errors.append("TrainCantidadStep: data_xgb_df es None.")
            return ctx

        X = data_xgb_df[XGB_FEATURES].copy()
        y = data_xgb_df[XGB_CANTIDAD_TARGET].fillna(0)

        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[CAT_FEATURES] = enc.fit_transform(X[CAT_FEATURES].fillna("DESCONOCIDO"))

        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)

        print(f"TrainCantidadStep: modelo entrenado con {len(X):,} muestras")

        model_xgb_cantidad = {
            "model": model,
            "encoder": enc,
            "features": XGB_FEATURES,
        }
        ctx.extra["model_xgb_cantidad"] = model_xgb_cantidad
        return ctx


class SaveModelStep(BaseStep[TrainingContext]):
    """
    Serializa todos los artefactos del pipeline en un único archivo .pkl
    usando joblib. El archivo contiene un dict con la clave "artefactos"
    que agrupa todos los modelos e infraestructura necesaria para predecir:

        - model_km: segmentación KMeans (no usada en predicción actualmente)
        - model_knn: vecinos cercanos con scaler, encoder y perfil de clientes
        - model_xgb_cantidad: regresor de cantidad con encoder y features
        - perfil_productos: historial completo de transacciones, usado en
          predicción para construir features de productos candidatos
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("SaveModelStep: clean_data es None.")
            return ctx

        artefactos = {
            "model_km": ctx.extra["model_km"],
            "model_knn": ctx.extra["model_knn"],
            "model_xgb_cantidad": ctx.extra["model_xgb_cantidad"],
            "model_apriori": ctx.extra["model_apriori"],
            "perfil_productos": ctx.clean_data,
        }

        MODEL_PATH_BASE.mkdir(parents=True, exist_ok=True)
        path_model = MODEL_PATH_BASE / f"modelo_{ctx.model_name}_{ctx.version}.pkl"
        joblib.dump({"artefactos": artefactos}, str(path_model))

        ctx.extra["path_model"] = path_model
        print(f"Modelo guardado en: {path_model}")
        return ctx


class RegistryModelStep(BaseStep[TrainingContext]):
    """
    Carga el modelo recién guardado desde disco y lo registra en el
    model_registry en memoria, dejándolo disponible para el pipeline
    de predicción sin necesidad de reiniciar el servidor.
    """

    def execute(self, ctx: TrainingContext) -> TrainingContext:
        path_model = str(ctx.extra.get("path_model"))

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
        return ctx
