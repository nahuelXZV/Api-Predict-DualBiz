from typing import cast

import pandas as pd
import numpy as np

from app.domain.ml.base_context import PredictContext
from app.domain.ml.base_step import BaseStep
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder

CANTIDAD_MINIMA = 1.0
TOP_N = 10


class LoadModelStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        art = ctx.model["artefactos"]

        # Infraestructura compartida
        ctx.extra["kmeans"] = art["kmeans"]
        ctx.extra["scaler"] = art["scaler"]
        ctx.extra["knn_segment"] = art["knn_segment"]
        ctx.extra["perfil_pivot"] = art["perfil_pivot"]
        ctx.extra["customer_profile"] = art["customer_profile"]
        ctx.extra["perfil_productos"] = art["perfil_productos"]
        ctx.extra["fecha_max"] = art["fecha_max"]

        # Modelo 1: cantidad
        ctx.extra["xgboost"] = art["xgboost"]
        ctx.extra["encoder"] = art["encoder"]
        ctx.extra["features"] = art["features"]

        # Modelo 2: afinidad
        ctx.extra["model_af"] = art["model_af"]
        ctx.extra["encoder_af"] = art["encoder_af"]
        ctx.extra["features_af"] = art["features_af"]

        return ctx


class ValidateClienteStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        if ctx.extra.get("customer_profile") is None:
            ctx.errors.append("ValidateClienteStep: modelo no cargado.")
            return ctx

        cliente_id = ctx.data.get("cliente_id")
        print(f"Validando cliente_id: {cliente_id}")
        if cliente_id is None:
            ctx.errors.append("ValidateClienteStep: 'cliente_id' no proporcionado.")
            return ctx

        customer_profile = cast(pd.DataFrame, ctx.extra["customer_profile"])
        existe = cliente_id in customer_profile["cliente_id"].values
        ctx.extra["cliente_es_nuevo"] = not existe

        print(f"Cliente {cliente_id} validado.")
        return ctx


class ResolveClientProfileStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        customer_profile = cast(pd.DataFrame, ctx.extra["customer_profile"])
        perfil_productos = cast(pd.DataFrame, ctx.extra["perfil_productos"])
        scaler = cast(StandardScaler, ctx.extra["scaler"])
        kmeans = ctx.extra["kmeans"]
        cliente_id = ctx.data.get("cliente_id")
        cliente_es_nuevo = ctx.extra["cliente_es_nuevo"]
        features_to_scale = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "meses_activo",
        ]

        if not cliente_es_nuevo:
            # ── Cliente existente: perfil real ────────────────────────────
            fila = customer_profile[customer_profile["cliente_id"] == cliente_id]
            segmento = int(fila["segmento"].iloc[0])
            perfil_vector = scaler.transform(fila[features_to_scale])

            ctx.extra["segmento"] = segmento
            ctx.extra["perfil_vector"] = perfil_vector
            print(f"Perfil resuelto (existente) → segmento {segmento}")
            return ctx

        # ── Cliente nuevo: construir perfil sintético ─────────────────────
        zona_id = ctx.data.get("zona_id")
        ruta_id = ctx.data.get("ruta_id")
        clasif = ctx.data.get("clasificacion_cliente")

        # Nivel 1: filtro completo zona + ruta + clasificación
        mascara = pd.Series(
            [True] * len(perfil_productos), index=perfil_productos.index
        )
        if zona_id is not None and "zona_id" in perfil_productos.columns:
            mascara &= perfil_productos["zona_id"] == zona_id
        if ruta_id is not None and "ruta_id" in perfil_productos.columns:
            mascara &= perfil_productos["ruta_id"] == ruta_id
        if clasif is not None and "clasificacion_cliente" in perfil_productos.columns:
            mascara &= perfil_productos["clasificacion_cliente"] == clasif

        clientes_ctx = perfil_productos.loc[mascara, "cliente_id"].unique()

        # Nivel 2: relajar a solo clasificación
        if len(clientes_ctx) == 0 and clasif is not None:
            # mascara_relajada = perfil_productos["clasificacion_cliente"] == clasif
            # clientes_ctx     = perfil_productos.loc[
            #     mascara_relajada, "cliente_id"
            # ].unique()
            print(f"Fallback nivel 2: solo clasificacion_cliente='{clasif}'")

        perfil_similares = customer_profile[
            customer_profile["cliente_id"].isin(clientes_ctx)
        ]

        if len(perfil_similares) > 0:
            # Perfil sintético = promedio de clientes similares
            perfil_sintetico = perfil_similares[features_to_scale].mean()
            x_sintetico = scaler.transform(
                pd.DataFrame([perfil_sintetico], columns=features_to_scale)
            )
            segmento = int(kmeans.predict(x_sintetico)[0])
            print(
                f"Perfil sintético construido con {len(perfil_similares)} clientes "
                f"similares → segmento {segmento}"
            )
        else:
            # Nivel 3: segmento mayoritario como último recurso
            segmento = int(customer_profile["segmento"].value_counts().idxmax())
            x_sintetico = kmeans.cluster_centers_[[segmento]]
            print(f"Fallback nivel 3: sin contexto → segmento mayoritario {segmento}")

        ctx.extra["segmento"] = segmento
        ctx.extra["perfil_vector"] = x_sintetico
        return ctx


class FindNeighborsStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        customer_profile = cast(pd.DataFrame, ctx.extra["customer_profile"])
        scaler = cast(StandardScaler, ctx.extra["scaler"])
        knn_segment = cast(dict, ctx.extra["knn_segment"])
        cliente_id = ctx.data.get("cliente_id")
        features_to_scale = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "meses_activo",
        ]

        fila = customer_profile[customer_profile["cliente_id"] == cliente_id]
        segmento = int(fila["segmento"].iloc[0])
        perfil_vector = scaler.transform(fila[features_to_scale])

        seg_data = knn_segment[segmento]
        knn_model = cast(NearestNeighbors, seg_data["model"])
        clientes_en_seg = seg_data["clientes"]

        distancias, indices = knn_model.kneighbors(perfil_vector)

        # Excluir índice 0 — siempre es el propio cliente
        vecinos_ids = [
            clientes_en_seg[i] for i in indices[0][1:] if i < len(clientes_en_seg)
        ]

        ctx.extra["segmento"] = segmento
        ctx.extra["vecinos_ids"] = vecinos_ids
        ctx.extra["vecinos_dist"] = distancias[0][1:].tolist()

        print(f"Segmento: {segmento} | Vecinos encontrados: {len(vecinos_ids)}")
        return ctx


class BuildCandidatesStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        perfil_pivot = cast(pd.DataFrame, ctx.extra["perfil_pivot"])
        vecinos_ids = cast(list, ctx.extra["vecinos_ids"])
        cliente_id = ctx.data.get("cliente_id")

        # Productos históricos del cliente
        if cliente_id in perfil_pivot.index:
            compras_cliente = perfil_pivot.loc[cliente_id]
            productos_propios = compras_cliente[compras_cliente > 0].index.tolist()
        else:
            productos_propios = []

        # Productos de los vecinos
        perfil_vecinos = perfil_pivot.loc[perfil_pivot.index.isin(vecinos_ids)]
        prom_vecinos_serie = (
            perfil_vecinos.mean(axis=0)
            if not perfil_vecinos.empty
            else pd.Series(dtype=float)
        )
        productos_vecinos = prom_vecinos_serie[prom_vecinos_serie > 0].index.tolist()
        todos_candidatos = list(set(productos_propios + productos_vecinos))
        # todos_candidatos = productos_vecinos

        ctx.extra["productos_propios"] = productos_propios
        ctx.extra["prom_vecinos_serie"] = prom_vecinos_serie
        ctx.extra["todos_candidatos"] = todos_candidatos

        print(
            f"Candidatos: {len(productos_propios)} propios | "
            f"{len(productos_vecinos)} de vecinos | "
            f"{len(todos_candidatos)} total único"
        )
        return ctx


class BuildFeatureMatrixStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        perfil_productos = cast(pd.DataFrame, ctx.extra["perfil_productos"])
        todos_candidatos = cast(list, ctx.extra["todos_candidatos"])
        prom_vecinos_serie = cast(pd.Series, ctx.extra["prom_vecinos_serie"])
        encoder = cast(OrdinalEncoder, ctx.extra["encoder"])
        encoder_af = cast(OrdinalEncoder, ctx.extra["encoder_af"])
        fecha_max = ctx.extra["fecha_max"]
        segmento = ctx.extra["segmento"]
        cliente_id = ctx.data.get("cliente_id")
        mes_actual = fecha_max.month

        historial_cliente = perfil_productos[
            perfil_productos["cliente_id"] == cliente_id
        ]
        ctx_base = historial_cliente.iloc[-1]

        filas = []
        for producto in todos_candidatos:
            hist_prod = historial_cliente[
                historial_cliente["nombre_producto"] == producto
            ]
            prod_info = perfil_productos[
                perfil_productos["nombre_producto"] == producto
            ]

            if len(hist_prod) > 0:
                ultima = hist_prod.sort_values("fecha_venta").iloc[-1]
                promedio_historico = hist_prod["cantidad_vendida"].mean()
                promedio_ultimas_3 = hist_prod["cantidad_vendida"].tail(3).mean()
                dias_entre_compras = hist_prod["dias_entre_compras"].mean()
                dias_desde_ultima_compra = (fecha_max - ultima["fecha_venta"]).days
                marca = ultima["marca"]
                linea_producto = ultima["linea_producto"]
                clasificacion_cliente = ultima["clasificacion_cliente"]
                ruta_id = ultima["ruta_id"]
                zona_id = ultima["zona_id"]
                sucursal = ultima["sucursal"]
                fuente = "historial_propio"
                num_compras_producto = len(hist_prod)
                meses_activo_producto = hist_prod["mes"].nunique()
                prom_mes = hist_prod[hist_prod["mes"] == mes_actual][
                    "cantidad_vendida"
                ].mean()
                prom_cantidad_mes = float(prom_mes) if pd.notna(prom_mes) else 0.0
                umbral_anio_ant = fecha_max - pd.Timedelta(days=300)
                compro_anio_ant = int(
                    len(
                        hist_prod[
                            (hist_prod["mes"] == mes_actual)
                            & (hist_prod["fecha_venta"] < umbral_anio_ant)
                        ]
                    )
                    > 0
                )
            else:
                prom_global = (
                    prod_info["cantidad_vendida"].mean() if len(prod_info) > 0 else 0.0
                )
                promedio_historico = (
                    float(prom_global) if pd.notna(prom_global) else 0.0
                )
                promedio_ultimas_3 = promedio_historico
                dias_entre_compras = float(ctx_base["dias_entre_compras"])
                dias_desde_ultima_compra = 999
                marca = (
                    prod_info["marca"].iloc[0]
                    if len(prod_info) > 0
                    else ctx_base["marca"]
                )
                linea_producto = (
                    prod_info["linea_producto"].iloc[0]
                    if len(prod_info) > 0
                    else ctx_base["linea_producto"]
                )
                clasificacion_cliente = ctx_base["clasificacion_cliente"]
                ruta_id = ctx_base["ruta_id"]
                zona_id = ctx_base["zona_id"]
                sucursal = ctx_base["sucursal"]
                fuente = "vecinos"
                num_compras_producto = 0
                meses_activo_producto = (
                    int(prod_info["mes"].nunique()) if len(prod_info) > 0 else 0
                )
                prom_mes = (
                    prod_info[prod_info["mes"] == mes_actual]["cantidad_vendida"].mean()
                    if len(prod_info) > 0
                    else np.nan
                )
                prom_cantidad_mes = float(prom_mes) if pd.notna(prom_mes) else 0.0
                compro_anio_ant = 0

            filas.append(
                {
                    "nombre_producto": producto,
                    "marca": marca,
                    "linea_producto": linea_producto,
                    "clasificacion_cliente": clasificacion_cliente,
                    "sucursal": sucursal,
                    "ruta_id": ruta_id,
                    "zona_id": zona_id,
                    "promedio_historico": promedio_historico,
                    "promedio_ultimas_3": promedio_ultimas_3,
                    "dias_entre_compras": dias_entre_compras,
                    "dias_desde_ultima_compra": dias_desde_ultima_compra,
                    "dia_semana": int(ctx_base["dia_semana"]),
                    "mes": int(ctx_base["mes"]),
                    "segmento": segmento,
                    "prom_vecinos": float(prom_vecinos_serie.get(producto, 0.0)),
                    "num_compras_producto": num_compras_producto,
                    "meses_activo_producto": meses_activo_producto,
                    "prom_cantidad_mes": prom_cantidad_mes,
                    "compro_este_mes_anio_ant": compro_anio_ant,
                    "_fuente": fuente,
                }
            )

        df_pred = pd.DataFrame(filas)

        cat_features = [
            "nombre_producto",
            "marca",
            "linea_producto",
            "clasificacion_cliente",
            "sucursal",
        ]

        df_cantidad = df_pred.copy()
        df_cantidad[cat_features] = encoder.transform(
            df_cantidad[cat_features].fillna("DESCONOCIDO")
        )

        df_afinidad = df_pred.copy()
        df_afinidad[cat_features] = encoder_af.transform(
            df_afinidad[cat_features].fillna("DESCONOCIDO")
        )

        ctx.extra["df_cantidad"] = df_cantidad
        ctx.extra["df_afinidad"] = df_afinidad
        ctx.extra["candidatos_raw"] = todos_candidatos

        print(f"Feature matrix lista: {len(df_pred)} productos candidatos")
        return ctx


class PredictStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        cantidad_minima = ctx.extra.get("cantidad_minima", CANTIDAD_MINIMA)
        top_n = ctx.extra.get("top_n", TOP_N)

        model_af = ctx.extra["model_af"]
        model_xgb = ctx.extra["xgboost"]
        features = ctx.extra["features"]
        features_af = ctx.extra["features_af"]

        df_cantidad = cast(pd.DataFrame, ctx.extra["df_cantidad"]).copy()
        df_afinidad = cast(pd.DataFrame, ctx.extra["df_afinidad"]).copy()
        candidatos_raw = ctx.extra["candidatos_raw"]

        # Paso 1: score de afinidad para todos los candidatos
        scores_af = np.clip(model_af.predict(df_afinidad[features_af]), 0, 1)

        df_cantidad["_nombre_raw"] = (
            candidatos_raw  # columna auxiliar, no entra al modelo
        )
        df_cantidad["score_afinidad"] = scores_af
        df_cantidad["_fuente"] = df_afinidad["_fuente"].values

        # Paso 2: top (top_n * 2) por afinidad
        top_af = df_cantidad.nlargest(top_n * 2, "score_afinidad").copy()

        # Paso 3: predecir cantidad — ahora nombre_producto sigue siendo el número encodeado
        top_af["cantidad_sugerida"] = np.maximum(
            model_xgb.predict(top_af[features]), 0
        ).round(2)

        # Paso 4: filtrar y ordenar
        df_result = top_af[top_af["cantidad_sugerida"] >= cantidad_minima].copy()

        df_result["_orden_fuente"] = (
            df_result["_fuente"]
            .map(
                {
                    "historial_propio": 0,
                    "vecinos": 1,
                }
            )
            .fillna(1)
            .astype(int)
        )

        df_result = df_result.sort_values(
            ["_orden_fuente", "score_afinidad"],
            ascending=[True, False],
        ).head(top_n)

        ctx.data_response = (
            df_result[
                [
                    "_nombre_raw",  # nombre legible
                    "score_afinidad",
                    "cantidad_sugerida",
                    "_fuente",
                ]
            ]
            .rename(columns={"_nombre_raw": "nombre_producto"})
            .copy()
        )
        return ctx
