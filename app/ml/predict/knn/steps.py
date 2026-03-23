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

        ctx.extra["kmeans"] = art["kmeans"]
        ctx.extra["scaler"] = art["scaler"]
        ctx.extra["knn_segment"] = art["knn_segment"]
        ctx.extra["perfil_pivot"] = art["perfil_pivot"]
        ctx.extra["xgboost"] = art["xgboost"]
        ctx.extra["encoder"] = art["encoder"]
        ctx.extra["features"] = art["features"]
        ctx.extra["customer_profile"] = art["customer_profile"]
        ctx.extra["perfil_productos"] = art["perfil_productos"]
        ctx.extra["fecha_max"] = art["fecha_max"]
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

        print(f"Cliente {cliente_id} existe en base: {existe}")
        if not existe:
            ctx.errors.append(
                f"ValidateClienteStep: cliente {cliente_id} no existe en la base."
            )
            return ctx

        print(f"Cliente {cliente_id} validado.")
        return ctx


class FindNeighborsStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        customer_profile = cast(pd.DataFrame, ctx.extra["customer_profile"])
        scaler = cast(StandardScaler, ctx.extra["scaler"])
        knn_segment = cast(dict, ctx.extra["knn_segment"])
        cliente_id = ctx.data.get("cliente_id")

        fila_cliente = customer_profile[customer_profile["cliente_id"] == cliente_id]
        segmento = int(fila_cliente["segmento"].iloc[0])
        ctx.extra["segmento"] = segmento

        seg_data = knn_segment[segmento]
        knn_model = cast(NearestNeighbors, seg_data["model"])
        clientes_en_seg = seg_data["clientes"]
        features_to_scale = [
            "cantidad_vendida",
            "promedio_historico",
            "dias_entre_compras",
            "meses_activo",
        ]

        # Posición del cliente dentro del segmento (mismo orden que en entrenamiento)
        perfil_ordenado = customer_profile.set_index("cliente_id").loc[clientes_en_seg]
        x_seg = scaler.transform(perfil_ordenado[features_to_scale])

        idx_local = clientes_en_seg.index(cliente_id)
        x_cliente = x_seg[idx_local].reshape(1, -1)

        distancias, indices = knn_model.kneighbors(x_cliente)

        # Excluir el primero (es el propio cliente) y convertir a IDs reales
        vecinos_locales = indices[0][1:]
        vecinos_dist = distancias[0][1:]

        vecinos_ids = [
            clientes_en_seg[i] for i in vecinos_locales if i < len(clientes_en_seg)
        ]

        ctx.extra["vecinos_ids"] = vecinos_ids
        ctx.extra["vecinos_dist"] = vecinos_dist.tolist()

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
        if not perfil_vecinos.empty:
            prom_vecinos_serie = perfil_vecinos.mean(axis=0)
        else:
            prom_vecinos_serie = pd.Series(dtype=float)

        productos_vecinos = prom_vecinos_serie[prom_vecinos_serie > 0].index.tolist()

        # Unión de candidatos: propios + de vecinos
        todos_candidatos = list(set(productos_propios + productos_vecinos))

        ctx.extra["productos_propios"] = productos_propios
        ctx.extra["prom_vecinos_serie"] = prom_vecinos_serie
        ctx.extra["todos_candidatos"] = todos_candidatos
        ctx.extra["perfil_vecinos"] = perfil_vecinos

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
        fecha_max = ctx.extra["fecha_max"]
        segmento = ctx.extra["segmento"]

        # Contexto del cliente: última fila de cada producto propio
        historial_cliente = perfil_productos[
            perfil_productos["cliente_id"] == ctx.data.get("cliente_id")
        ]

        # Contexto base del cliente (para productos nuevos sin historial propio)
        if len(historial_cliente) > 0:
            ctx_base = historial_cliente.iloc[-1]
        else:
            ctx_base = perfil_productos.iloc[-1]  # fallback a cualquier fila

        filas = []
        for producto in todos_candidatos:
            hist_prod = historial_cliente[
                historial_cliente["nombre_producto"] == producto
            ]

            if len(hist_prod) > 0:
                # Producto con historial propio: usar sus datos reales
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
            else:
                # Producto nuevo para este cliente: usar contexto del cliente
                # y promedio global del producto como referencia
                prom_global = perfil_productos[
                    perfil_productos["nombre_producto"] == producto
                ]["cantidad_vendida"].mean()

                promedio_historico = prom_global if pd.notna(prom_global) else 0.0
                promedio_ultimas_3 = promedio_historico
                dias_entre_compras = float(ctx_base["dias_entre_compras"])
                dias_desde_ultima_compra = 999  # nunca lo ha comprado
                marca = (
                    perfil_productos[perfil_productos["nombre_producto"] == producto][
                        "marca"
                    ].iloc[0]
                    if len(
                        perfil_productos[
                            perfil_productos["nombre_producto"] == producto
                        ]
                    )
                    > 0
                    else ctx_base["marca"]
                )
                linea_producto = (
                    perfil_productos[perfil_productos["nombre_producto"] == producto][
                        "linea_producto"
                    ].iloc[0]
                    if len(
                        perfil_productos[
                            perfil_productos["nombre_producto"] == producto
                        ]
                    )
                    > 0
                    else ctx_base["linea_producto"]
                )
                clasificacion_cliente = ctx_base["clasificacion_cliente"]
                ruta_id = ctx_base["ruta_id"]
                zona_id = ctx_base["zona_id"]
                sucursal = ctx_base["sucursal"]
                fuente = "vecinos"

            prom_vecinos = float(prom_vecinos_serie.get(producto, 0.0))

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
                    "prom_vecinos": prom_vecinos,
                    "_fuente": fuente,  # columna auxiliar, no entra al modelo
                }
            )

        df_pred = pd.DataFrame(filas)

        # Encodear categóricas con el mismo encoder del entrenamiento
        cat_features = [
            "nombre_producto",
            "marca",
            "linea_producto",
            "clasificacion_cliente",
            "sucursal",
        ]
        df_pred[cat_features] = encoder.transform(
            df_pred[cat_features].fillna("DESCONOCIDO")
        )

        ctx.extra["df_pred"] = df_pred
        ctx.extra["candidatos_raw"] = (
            todos_candidatos  # nombres originales antes del encode
        )

        print(f"Feature matrix lista: {len(df_pred)} productos candidatos")
        return ctx


class PredictStep(BaseStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        cantidad_minima = ctx.extra.get("cantidad_minima", CANTIDAD_MINIMA)
        top_n = ctx.extra.get("top_n", TOP_N)

        modelo = ctx.extra["xgboost"]
        features = ctx.extra["features"]
        df_pred = cast(pd.DataFrame, ctx.extra["df_pred"]).copy()

        X = df_pred[features]
        cantidades_pred = modelo.predict(X)

        # Reconstruir con nombres originales (antes del encode)
        df_pred["nombre_producto_raw"] = ctx.extra["candidatos_raw"]
        df_pred["cantidad_sugerida"] = np.maximum(cantidades_pred, 0)
        df_pred["cantidad_sugerida"] = df_pred["cantidad_sugerida"].round(1)

        # Filtrar productos con cantidad significativa
        df_result = df_pred[df_pred["cantidad_sugerida"] >= cantidad_minima].copy()

        # Ordenar: primero los de historial propio, luego por cantidad descendente
        df_result["_orden_fuente"] = (df_result["_fuente"] == "vecinos").astype(int)
        print("productos antes de filtrar")
        print(df_result.info())
        print(df_result.head(20))
        df_result = df_result.sort_values(
            ["_orden_fuente", "cantidad_sugerida"],
            ascending=[True, False],
        ).head(top_n)

        print(f"Predicción lista: {len(df_result)} productos sugeridos")
        print(df_result[["nombre_producto_raw", "cantidad_sugerida", "_fuente"]])
        ctx.data_response = df_result[
            ["nombre_producto_raw", "cantidad_sugerida", "_fuente"]
        ].copy()
        return ctx
