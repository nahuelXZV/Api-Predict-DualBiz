import pandas as pd
import numpy as np

from app.domain.ml.base_context import PredictContext
from app.domain.ml.base_step import BaseStep

CANTIDAD_MINIMA = 1.0
TOP_N = 10

CAT_FEATURES = [
    "nombre_producto",
    "marca",
    "linea_producto",
    "clasificacion_cliente",
    "sucursal",
]


class LoadModelStep(BaseStep[PredictContext]):
    """
    Desempaqueta los artefactos del modelo cargado en memoria y los expone
    en ctx.extra para que los pasos siguientes puedan acceder a ellos:
        - model_knn: vecinos cercanos (incluye scaler, encoder, perfil de clientes)
        - model_xgb_cantidad: regresor de cantidad (incluye encoder y features)
        - perfil_productos: historial completo de transacciones del entrenamiento
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        art = ctx.model["artefactos"]
        ctx.extra["model_knn"] = art["model_knn"]
        ctx.extra["model_xgb_cantidad"] = art["model_xgb_cantidad"]
        ctx.extra["perfil_productos"] = art["perfil_productos"]
        return ctx


class ValidateClienteStep(BaseStep[PredictContext]):
    """
    Valida que el request sea procesable antes de continuar:
        - Verifica que el modelo esté cargado.
        - Verifica que se haya proporcionado cliente_id.
        - Verifica que el cliente tenga historial en el perfil KNN. Si no
          existe, no se puede construir su vector de similitud ni encontrar
          vecinos, por lo que el pipeline se detiene con un error descriptivo.
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        if ctx.extra.get("model_knn") is None:
            ctx.errors.append("ValidateClienteStep: modelo no cargado.")
            return ctx

        cliente_id = ctx.data.get("cliente_id")
        if cliente_id is None:
            ctx.errors.append("ValidateClienteStep: 'cliente_id' no proporcionado.")
            return ctx

        customer_profile = ctx.extra["model_knn"]["data"]
        if cliente_id not in customer_profile["cliente_id"].values:
            ctx.errors.append(
                f"ValidateClienteStep: cliente {cliente_id} no tiene historial."
            )
            return ctx

        print(f"Cliente {cliente_id} validado.")
        return ctx


class FindNeighborsStep(BaseStep[PredictContext]):
    """
    Encuentra los clientes más similares al cliente consultado usando el
    modelo KNN entrenado con distancia coseno.

    El cliente se representa con el mismo vector usado en entrenamiento:
    features numéricas escaladas con StandardScaler + features categóricas
    codificadas con OneHotEncoder. Se excluye el propio cliente de los
    resultados.

    Guarda en ctx.extra:
        - segmento: segmento KMeans del cliente (feature para el regresor)
        - vecinos_ids: lista de cliente_id de los vecinos encontrados
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        model_knn = ctx.extra["model_knn"]
        cliente_id = ctx.data.get("cliente_id")

        customer_profile = model_knn["data"]
        scaler = model_knn["scaler"]
        enc_cat = model_knn["enc_cat"]
        feats_num = model_knn["feats_num"]
        cat_features = model_knn["cat_features"]
        knn = model_knn["model"]
        all_customers = model_knn["customers"]

        fila = customer_profile[customer_profile["cliente_id"] == cliente_id]
        segmento = int(fila["segmento"].iloc[0])

        x_num = scaler.transform(fila[feats_num])
        x_cat = enc_cat.transform(fila[cat_features].astype(str))
        perfil_vector = np.hstack([x_num, x_cat])

        _, indices = knn.kneighbors(perfil_vector)

        vecinos_ids = [
            all_customers[i]
            for i in indices[0]
            if i < len(all_customers) and all_customers[i] != cliente_id
        ]

        ctx.extra["segmento"] = segmento
        ctx.extra["vecinos_ids"] = vecinos_ids
        print(f"Segmento: {segmento} | Vecinos: {len(vecinos_ids)}")
        return ctx


class BuildCandidatesStep(BaseStep[PredictContext]):
    """
    Construye la lista de productos candidatos a recomendar a partir del
    historial de compras de los vecinos.

    Construye una matriz pivot (cliente × producto) desde perfil_productos
    y filtra solo las filas de los vecinos. Con esa submatriz calcula:
        - prom_vecinos: cantidad promedio que compran los vecinos por producto
        - pct_vecinos: fracción de vecinos que compran cada producto (score)

    Si solo_nuevos=True (default), excluye los productos que el cliente ya
    compra para enfocarse en descubrimiento de productos nuevos.

    Guarda en ctx.extra:
        - candidatos: lista de productos a evaluar
        - pct_vecinos: Serie indexada por nombre_producto con el score
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        perfil_productos = ctx.extra["perfil_productos"]
        vecinos_ids = ctx.extra["vecinos_ids"]

        perfil_pivot = perfil_productos.pivot_table(
            index="cliente_id",
            columns="nombre_producto",
            values="cantidad_vendida",
            aggfunc="sum",
            fill_value=0,
        )

        perfil_vecinos = perfil_pivot.loc[perfil_pivot.index.isin(vecinos_ids)]
        prom_vecinos = (
            perfil_vecinos.mean(axis=0)
            if not perfil_vecinos.empty
            else pd.Series(dtype=float)
        )
        pct_vecinos = (
            (perfil_vecinos > 0).mean(axis=0)
            if not perfil_vecinos.empty
            else pd.Series(dtype=float)
        )
        candidatos_vecinos = prom_vecinos[prom_vecinos > 0].index.tolist()

        solo_nuevos = ctx.data.get("solo_nuevos", True)
        if solo_nuevos:
            cliente_id = ctx.data.get("cliente_id")
            productos_propios = set(
                perfil_productos[perfil_productos["cliente_id"] == cliente_id][
                    "nombre_producto"
                ].unique()
            )
            candidatos = [p for p in candidatos_vecinos if p not in productos_propios]
            print(
                f"Candidatos vecinos: {len(candidatos_vecinos)} | Nuevos para el cliente: {len(candidatos)}"
            )
        else:
            candidatos = candidatos_vecinos
            print(f"Candidatos vecinos: {len(candidatos)} (incluyendo propios)")

        ctx.extra["candidatos"] = candidatos
        ctx.extra["pct_vecinos"] = pct_vecinos
        return ctx


class BuildFeaturesStep(BaseStep[PredictContext]):
    """
    Construye el DataFrame de features para cada producto candidato,
    en el mismo formato que el modelo XGBRegressor espera.

    Para cada producto distingue dos casos:
        - historial_propio: el cliente ya compró el producto alguna vez.
          Se usan sus propios promedios, recencia y datos de la última compra.
        - vecinos: el cliente nunca compró el producto. Se usan los promedios
          globales del producto y los datos base del cliente (sucursal, ruta,
          zona, clasificación) como contexto.

    La fecha_max se obtiene desde perfil_productos para calcular recencia.
    El mes actual se usa como señal de estacionalidad.

    Guarda en ctx.extra:
        - df_features: DataFrame listo para predicción con XGBRegressor
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        perfil_productos = ctx.extra["perfil_productos"]
        candidatos = ctx.extra["candidatos"]
        segmento = ctx.extra["segmento"]
        cliente_id = ctx.data.get("cliente_id")

        fecha_max = perfil_productos["fecha_venta"].max()
        mes_actual = fecha_max.month

        historial_cliente = perfil_productos[
            perfil_productos["cliente_id"] == cliente_id
        ]
        ctx_base = historial_cliente.sort_values("fecha_venta").iloc[-1]

        filas = []
        for producto in candidatos:
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
                fuente = "historial_propio"
            else:
                promedio_historico = (
                    float(prod_info["cantidad_vendida"].mean())
                    if len(prod_info) > 0
                    else 0.0
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
                fuente = "vecinos"

            filas.append(
                {
                    "nombre_producto": producto,
                    "marca": marca,
                    "linea_producto": linea_producto,
                    "clasificacion_cliente": ctx_base["clasificacion_cliente"],
                    "sucursal": ctx_base["sucursal"],
                    "ruta_id": ctx_base["ruta_id"],
                    "zona_id": ctx_base["zona_id"],
                    "promedio_historico": promedio_historico,
                    "promedio_ultimas_3": promedio_ultimas_3,
                    "dias_entre_compras": dias_entre_compras,
                    "dias_desde_ultima_compra": dias_desde_ultima_compra,
                    "dia_semana": int(ctx_base["dia_semana"]),
                    "mes": mes_actual,
                    "segmento": segmento,
                    "_fuente": fuente,
                }
            )

        ctx.extra["df_features"] = pd.DataFrame(filas)
        print(f"Feature matrix: {len(filas)} productos candidatos")
        return ctx


class RankAndPredictStep(BaseStep[PredictContext]):
    """
    Genera las recomendaciones finales combinando el score de vecinos con
    la predicción de cantidad del XGBRegressor.

    Parámetros configurables vía ctx.extra:
        - top_n (default 10): número máximo de productos a retornar
        - cantidad_minima (default 1.0): filtra productos con cantidad
          sugerida menor a este valor

    Flujo:
        1. Asigna score a cada candidato = pct_vecinos (fracción de vecinos
           que compran ese producto). A mayor score, más recomendado.
        2. Pre-selecciona top Nx3 por score para reducir el costo de
           inferencia del regresor.
        3. Predice la cantidad sugerida con XGBRegressor solo para los
           pre-seleccionados.
        4. Filtra por cantidad_minima y retorna top N ordenados por score.

    Respuesta en ctx.data_response: nombre_producto, cantidad_sugerida,
    score, fuente (vecinos / historial_propio).
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        top_n = ctx.extra.get("top_n", TOP_N)
        cantidad_minima = ctx.extra.get("cantidad_minima", CANTIDAD_MINIMA)

        model_xgb_cantidad = ctx.extra["model_xgb_cantidad"]
        pct_vecinos = ctx.extra["pct_vecinos"]
        features = model_xgb_cantidad["features"]

        df = ctx.extra["df_features"].copy()

        # Paso 1: score = % de vecinos que compran el producto
        df["score"] = df["nombre_producto"].map(pct_vecinos).fillna(0)

        # Paso 2: top N*3 por score antes de predecir cantidad
        top_df = df.nlargest(top_n * 3, "score").copy()

        # Paso 3: predecir cantidad solo para los pre-seleccionados
        df_cantidad = top_df.copy()
        df_cantidad[CAT_FEATURES] = model_xgb_cantidad["encoder"].transform(
            df_cantidad[CAT_FEATURES].fillna("DESCONOCIDO")
        )
        top_df["cantidad_sugerida"] = np.maximum(
            model_xgb_cantidad["model"].predict(df_cantidad[features]), 0
        ).round(2)

        # Paso 4: filtrar por cantidad mínima y tomar top N
        resultado = top_df[top_df["cantidad_sugerida"] >= cantidad_minima]
        resultado = resultado.nlargest(top_n, "score")

        ctx.data_response = resultado[
            [
                "nombre_producto",
                "cantidad_sugerida",
                "score",
                "_fuente",
            ]
        ].rename(columns={"_fuente": "fuente"})
        return ctx
