from typing import cast

import joblib
import pandas as pd
import xgboost as xgb
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

from app.domain.ml.base_step import BaseTrainingStep
from app.domain.ml.base_context import TrainingContext
from app.domain.ml.model_metadata import ModelMetadata
from app.domain.ml.model_registry import model_registry
from app.ml.models.knn_model import KnnModel

BASE_DIR = Path(__file__).resolve().parents[4]
DATA_PATH = BASE_DIR / "storage" / "data" / "consulta_base.csv"
MODEL_PATH_BASE = BASE_DIR / "storage" / "models"


class LoadDataStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        df = pd.read_csv(DATA_PATH)
        print("Dataset original")
        df.info()
        print(df.head())

        ctx.raw_data = df
        return ctx


class CleanDataStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.raw_data is None:
            ctx.errors.append(
                "CleanDataStep: raw_data es None, LoadDataStep no se ejecutó."
            )
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

        df = df.dropna(subset=["cliente_id", "nombre_producto", "cantidad_vendida"])
        df = df[df["cantidad_vendida"] > 0]

        ctx.clean_data = df
        print("Dataset limpio:")
        print(df.info())
        print(df.head())
        return ctx


class AddDerivedFeatureStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append(
                "AddDerivedFeatureStep: clean_data es None, CleanDataStep no se ejecutó."
            )
            return ctx

        df = ctx.clean_data.copy()

        df["fecha_venta"] = pd.to_datetime(df["fecha_venta"])
        df = df.sort_values(by=["cliente_id", "nombre_producto", "fecha_venta"])

        # 4. Cálculo de Features (Frecuencia, Promedios, Recencia)
        df["fecha_anterior"] = df.groupby(["cliente_id", "nombre_producto"])[
            "fecha_venta"
        ].shift(1)
        df["dias_entre_compras"] = (df["fecha_venta"] - df["fecha_anterior"]).dt.days
        df["promedio_historico"] = df.groupby(["cliente_id", "nombre_producto"])[
            "cantidad_vendida"
        ].transform("mean")

        fecha_maxima = df["fecha_venta"].max()
        ultima_compra = df.groupby(["cliente_id", "nombre_producto"])[
            "fecha_venta"
        ].transform("max")
        df["dias_desde_ultima_compra"] = (fecha_maxima - ultima_compra).dt.days

        # 5. Estacionalidad
        df["dia_semana"] = df["fecha_venta"].dt.dayofweek
        df["mes"] = df["fecha_venta"].dt.month

        # 6. Limpieza y Exportación
        df["dias_entre_compras"] = df["dias_entre_compras"].fillna(0)

        ctx.clean_data = df
        print("Dataset con features derivados:")
        print(df.head())
        return ctx


class KMeansStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append(
                "KMeansStep: clean_data es None, AddDerivedFeatureStep no se ejecutó."
            )
            return ctx

        df = ctx.clean_data.copy()
        customer_profile_df = (
            df.groupby("cliente_id")
            .agg(
                {
                    "cantidad_vendida": "sum",  # Volumen total
                    "promedio_historico": "mean",  # Tamaño de pedido habitual
                    "dias_entre_compras": "mean",  # Frecuencia de visita
                    "mes": "nunique",  # Constancia a lo largo del año
                }
            )
            .reset_index()
            .fillna(0)
        )
        print("Perfil de clientes antes de segmentar:")
        print(customer_profile_df.head())
        print(customer_profile_df.info())

        scaler = StandardScaler()
        features_to_scale = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "mes",
        ]

        x_km = scaler.fit_transform(customer_profile_df[features_to_scale])
        kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42, n_init=10)
        segmento = kmeans.fit_predict(x_km)
        customer_profile_df["segmento"] = segmento

        ctx.extra["customer_profile"] = customer_profile_df
        ctx.extra["kmeans_model"] = kmeans
        ctx.extra["scaler"] = scaler
        ctx.extra["features_to_scale"] = features_to_scale

        print("Datos Segmentados:")
        print(customer_profile_df.head())
        print(customer_profile_df.info())
        return ctx


class KnnStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append(
                "KnnStep: clean_data es None, AddDerivedFeatureStep no se ejecutó."
            )
            return ctx
        df = ctx.clean_data.copy()
        scaler = cast(StandardScaler, ctx.extra.get("scaler"))
        features_to_scale = ctx.extra.get("features_to_scale")

        customer_profile_df = cast(
            pd.DataFrame, ctx.extra.get("customer_profile")
        ).copy()

        perfil_pivot = (
            df.groupby(["cliente_id", "nombre_producto"])["cantidad_vendida"]
            .sum()
            .unstack(fill_value=0)
        )

        knn_segment = {}

        for segmento in customer_profile_df["segmento"].unique():
            idx = customer_profile_df[customer_profile_df["segmento"] == segmento].index
            x_knn = scaler.transform(customer_profile_df.loc[idx, features_to_scale])
            knn = NearestNeighbors(n_neighbors=10, metric="cosine")
            knn.fit(x_knn)
            clientes_seg = cast(
                pd.Series, customer_profile_df.loc[idx, "cliente_id"]
            ).to_numpy()
            knn_segment[segmento] = {
                "model": knn,
                "index": idx,
                "perfil": perfil_pivot.loc[perfil_pivot.index.isin(clientes_seg)],
            }

        ctx.extra["knn_segment"] = knn_segment

        return ctx


class PrepareDataXGBoostStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append(
                "PrepareDataXGBoostStep: clean_data es None, AddDerivedFeatureStep no se ejecutó."
            )
            return ctx

        df = ctx.clean_data.copy()
        customer_profile_df = cast(
            pd.DataFrame, ctx.extra.get("customer_profile")
        ).copy()
        knn_segment = cast(dict, ctx.extra.get("knn_segment"))
        scaler = cast(StandardScaler, ctx.extra.get("scaler"))
        features_to_scale = ctx.extra.get("features_to_scale")

        data_xgb_df = (
            df.groupby(["cliente_id", "nombre_producto"])
            .agg(
                {
                    "cantidad_vendida": "sum",
                    "promedio_historico": "mean",
                    "dias_entre_compras": "mean",
                    "dias_desde_ultima_compra": "mean",
                    "dia_semana": "first",
                    "mes": "first",
                    "marca": "first",
                    "linea_producto": "first",
                    "clasificacion_cliente": "first",
                    "ruta_id": "first",
                    "zona_id": "first",
                    "sucursal": "first",
                }
            )
            .reset_index()
        )

        seg_map = customer_profile_df.set_index("cliente_id")["segmento"]
        data_xgb_df["segmento"] = data_xgb_df["cliente_id"].map(seg_map)

        def prom_vecinos(row: pd.Series) -> float:
            seg = str(row["segmento"])
            data = knn_segment[seg]
            cli = cast(str, row["cliente_id"])
            prod = cast(str, row["nombre_producto"])

            mask = customer_profile_df["cliente_id"] == cli
            x_knn = customer_profile_df.loc[mask, features_to_scale]

            if x_knn.empty:
                return 0.0

            x_knn = scaler.transform(cast(pd.DataFrame, x_knn))
            ids = data["model"].kneighbors(x_knn, return_distance=False)[0][1:]

            vecinos = cast(pd.DataFrame, data["perfil"]).iloc[ids]

            if prod not in vecinos.columns:
                return 0.0

            return float(cast(pd.Series, vecinos[prod]).mean())

        data_xgb_df["prom_vecinos"] = data_xgb_df.apply(prom_vecinos, axis=1)

        cat_features = [
            "marca",
            "clasificacion_cliente",
            "linea_producto",
            "sucursal",
        ]
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        data_xgb_df[cat_features] = enc.fit_transform(data_xgb_df[cat_features])

        ctx.extra["data_xgb_df"] = data_xgb_df
        ctx.extra["encoder"] = enc

        return ctx


class SaveModelStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append(
                "SaveModelStep: clean_data es None, CleanDataStep no se ejecutó."
            )
            return ctx

        data_xgb_df = cast(pd.DataFrame, ctx.extra.get("data_xgb_df")).copy()
        knn_segment = cast(dict, ctx.extra.get("knn_segment"))
        scaler = cast(StandardScaler, ctx.extra.get("scaler"))
        k_means = cast(KMeans, ctx.extra.get("kmeans_model"))
        encoder = cast(OrdinalEncoder, ctx.extra.get("encoder"))

        FEATURES = [
            "promedio_historico",
            "dias_entre_compras",
            "dias_desde_ultima_compra",
            "dia_semana",
            "mes",
            "segmento",
            "prom_vecinos",
            "marca",
            "linea_producto",
            "clasificacion_cliente",
            "ruta_id",
            "zona_id",
            "sucursal",
        ]

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        x = data_xgb_df[FEATURES]
        # y = data_xgb_df["cantidad_vendida"].fillna(0)
        y = (data_xgb_df["cantidad_vendida"] > 0).astype(int)
        model.fit(x, y)

        artefactos = {
            "kmeans": k_means,
            "scaler": scaler,
            "knn_segment": knn_segment,
            "xgboost": model,
            "encoder": encoder,
            "perfil_feats": data_xgb_df,
            "perfil_productos": ctx.clean_data,
            "features": FEATURES,
        }

        path_model = MODEL_PATH_BASE / f"modelo_{ctx.model_name}_{ctx.version}.pkl"
        ctx.extra["path_model"] = path_model

        joblib.dump({"artefactos": artefactos}, str(path_model))
        print(f"Modelo guardado en {path_model}")
        return ctx


class RegistryModelStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        path_model = str(ctx.extra.get("path_model"))

        meta_data = ModelMetadata(
            name=ctx.model_name,
            version=ctx.version,
            path_model=path_model,
        )
        model = KnnModel(metadata=meta_data)
        model.load(path_model)

        model_registry.register(
            name=ctx.model_name,
            model=model,
        )
        return ctx
