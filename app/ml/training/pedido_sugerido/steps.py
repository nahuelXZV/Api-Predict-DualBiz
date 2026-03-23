from typing import cast

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

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
    "prom_vecinos",  # cuánto compran clientes similares este producto
]

XGB_TARGET = "cantidad_vendida"


class LoadDataStep(BaseStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        df = pd.read_csv(DATA_PATH)
        print(f"Dataset original: {len(df):,} filas | {df.shape[1]} columnas")
        ctx.raw_data = df
        return ctx


class CleanDataStep(BaseStep[TrainingContext]):
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
        df = df[df["cantidad_vendida"] > 0]
        print(f"Limpieza: {antes - len(df):,} filas eliminadas → {len(df):,} útiles")

        ctx.clean_data = df
        return ctx


class AddDerivedFeatureStep(BaseStep[TrainingContext]):
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
        ctx.extra["fecha_max"] = fecha_max
        print(f"Features derivadas calculadas. Columnas: {list(df.columns)}")
        return ctx


class KMeansStep(BaseStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("KMeansStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()
        customer_profile_df = (
            df.groupby("cliente_id")
            .agg(
                cantidad_vendida=("cantidad_vendida", "sum"),
                promedio_historico=("promedio_historico", "mean"),
                dias_entre_compras=("dias_entre_compras", "mean"),
                meses_activo=("mes", "nunique"),
            )
            .reset_index()
            .fillna(0)
        )

        print("Perfil de clientes antes de segmentar:")
        print(customer_profile_df.head())
        print(customer_profile_df.info())

        features_to_scale = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "meses_activo",
        ]
        scaler = StandardScaler()
        x_km = scaler.fit_transform(customer_profile_df[features_to_scale])

        kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
        customer_profile_df["segmento"] = kmeans.fit_predict(x_km)

        print(f"Clustering: {len(customer_profile_df):,} clientes en 5 segmentos")
        print(customer_profile_df["segmento"].value_counts().sort_index())

        ctx.extra["customer_profile"] = customer_profile_df
        ctx.extra["kmeans_model"] = kmeans
        ctx.extra["scaler"] = scaler
        ctx.extra["features_to_scale"] = features_to_scale
        ctx.extra["x_scaled"] = x_km

        print("Datos Segmentados:")
        print(customer_profile_df.head())
        print(customer_profile_df.info())
        return ctx


class KnnStep(BaseStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("KnnStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()
        scaler = cast(StandardScaler, ctx.extra.get("scaler"))
        features_to_scale = ctx.extra.get("features_to_scale")
        customer_profile_df = cast(pd.DataFrame, ctx.extra["customer_profile"]).copy()

        perfil_pivot = (
            df.groupby(["cliente_id", "nombre_producto"])["cantidad_vendida"]
            .sum()
            .unstack(fill_value=0)
        )

        print("Perfil pivot de clientes-productos:")
        print(perfil_pivot.info())
        print(perfil_pivot.head())

        knn_segment: dict = {}

        for segmento in sorted(customer_profile_df["segmento"].unique()):
            print(f"Construyendo KNN para segmento {segmento}")
            mask_seg = customer_profile_df["segmento"] == segmento
            clientes_seg = cast(
                pd.Series, customer_profile_df.loc[mask_seg, "cliente_id"]
            ).to_list()

            x_knn = scaler.transform(
                customer_profile_df.loc[mask_seg, features_to_scale]
            )
            n_neighbors = min(11, len(clientes_seg))

            knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
            knn.fit(x_knn)

            perfil_seg = perfil_pivot.loc[perfil_pivot.index.isin(clientes_seg)].copy()

            knn_segment[int(segmento)] = {
                "model": knn,
                "clientes": clientes_seg,
                "perfil": perfil_seg,
                "cliente_to_idx": {c: i for i, c in enumerate(clientes_seg)},
            }

        ctx.extra["knn_segment"] = knn_segment
        ctx.extra["perfil_pivot"] = perfil_pivot
        return ctx


class PrepareDataXGBoostStep(BaseStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("PrepareDataXGBoostStep: clean_data es None.")
            return ctx

        df = ctx.clean_data.copy()
        customer_profile_df = cast(pd.DataFrame, ctx.extra["customer_profile"]).copy()
        knn_segment = cast(dict, ctx.extra["knn_segment"])
        scaler = cast(StandardScaler, ctx.extra["scaler"])
        features_to_scale = ctx.extra["features_to_scale"]

        # Agregar una fila por cliente-producto (nivel de predicción)
        data_xgb_df = (
            df.groupby(["cliente_id", "nombre_producto"])
            .agg(
                cantidad_vendida=("cantidad_vendida", "sum"),
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
            )
            .reset_index()
        )

        seg_map = customer_profile_df.set_index("cliente_id")["segmento"]
        data_xgb_df["segmento"] = (
            data_xgb_df["cliente_id"].map(seg_map).fillna(-1).astype(int)
        )

        print("Calculando prom_vecinos por segmento (vectorizado)...")
        prom_vecinos_map: dict = {}

        for seg_id, seg_data in knn_segment.items():
            clientes_en_seg = seg_data["clientes"]
            perfil_seg = cast(pd.DataFrame, seg_data["perfil"])
            knn_model = cast(NearestNeighbors, seg_data["model"])

            # Perfiles de comportamiento de clientes en este segmento
            perfil_ordenado = customer_profile_df.set_index("cliente_id").loc[
                clientes_en_seg
            ]
            x_seg = scaler.transform(perfil_ordenado[features_to_scale])

            # Vecinos para todos los clientes del segmento de una sola vez
            _, indices_matrix = knn_model.kneighbors(x_seg)  # shape: (n_clientes, k)

            for local_idx, cliente_id in enumerate(clientes_en_seg):
                # índices de vecinos (excluimos el primero: es el propio cliente)
                vecino_local_idxs = indices_matrix[local_idx][1:]
                vecinos_ids = [
                    clientes_en_seg[i]
                    for i in vecino_local_idxs
                    if i < len(clientes_en_seg)
                ]

                if len(vecinos_ids) == 0:
                    continue

                # Promedio de cantidad que esos vecinos compran por producto
                perfil_vecinos = perfil_seg.loc[perfil_seg.index.isin(vecinos_ids)]
                if perfil_vecinos.empty:
                    continue

                prom_por_producto = perfil_vecinos.mean(axis=0)

                for prod, prom in prom_por_producto.items():
                    prom_vecinos_map[(cliente_id, prod)] = float(prom)

        data_xgb_df["prom_vecinos"] = data_xgb_df.apply(
            lambda r: prom_vecinos_map.get(
                (r["cliente_id"], r["nombre_producto"]), 0.0
            ),
            axis=1,
        )
        print(
            f"prom_vecinos calculado. Cobertura: "
            f"{(data_xgb_df['prom_vecinos'] > 0).mean() * 100:.1f}% de filas"
        )

        # ── Encodear variables categóricas ───────────────────────────────
        cat_features = [
            "nombre_producto",
            "marca",
            "linea_producto",
            "clasificacion_cliente",
            "sucursal",
        ]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        data_xgb_df[cat_features] = enc.fit_transform(
            data_xgb_df[cat_features].fillna("DESCONOCIDO")
        )

        ctx.extra["data_xgb_df"] = data_xgb_df
        ctx.extra["encoder"] = enc
        print(f"Dataset XGBoost listo: {len(data_xgb_df):,} filas")
        return ctx


class TrainXGBoostStep(BaseStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        data_xgb_df = cast(pd.DataFrame, ctx.extra.get("data_xgb_df")).copy()
        if data_xgb_df is None:
            ctx.errors.append("TrainXGBoostStep: data_xgb_df es None.")
            return ctx

        X = data_xgb_df[XGB_FEATURES]
        y = data_xgb_df[XGB_TARGET].fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # XGBRegressor: predice CUÁNTO comprar, no solo si compra o no
        model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        mae = mean_absolute_error(y_test, model.predict(X_test))
        print(f"XGBoost entrenado | MAE en test: {mae:.2f} unidades")

        ctx.extra["xgboost_model"] = model
        ctx.extra["xgb_features"] = XGB_FEATURES
        ctx.extra["mae"] = mae
        return ctx


class SaveModelStep(BaseStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("SaveModelStep: clean_data es None.")
            return ctx

        artefactos = {
            "kmeans": ctx.extra["kmeans_model"],
            "scaler": ctx.extra["scaler"],
            "knn_segment": ctx.extra["knn_segment"],
            "perfil_pivot": ctx.extra["perfil_pivot"],
            "xgboost": ctx.extra["xgboost_model"],
            "encoder": ctx.extra["encoder"],
            "features": ctx.extra["xgb_features"],
            "customer_profile": ctx.extra["customer_profile"],
            "perfil_productos": ctx.clean_data,
            "fecha_max": ctx.extra["fecha_max"],
            "mae": ctx.extra["mae"],
        }

        MODEL_PATH_BASE.mkdir(parents=True, exist_ok=True)
        path_model = MODEL_PATH_BASE / f"modelo_{ctx.model_name}_{ctx.version}.pkl"
        joblib.dump({"artefactos": artefactos}, str(path_model))

        ctx.extra["path_model"] = path_model
        print(f"Modelo guardado en: {path_model}")
        print(f"MAE final: {ctx.extra['mae']:.2f} unidades")
        return ctx


class RegistryModelStep(BaseStep[TrainingContext]):
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
