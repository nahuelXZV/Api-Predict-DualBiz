import pandas as pd
import numpy as np

from app.domain.core.logging import logger
from app.domain.ml.base_context import PredictContext
from app.domain.ml.base_step import BaseStep
from app.ml.predict.pedido_sugerido.utils import build_features_candidatos

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
        - model_apriori: reglas de asociación entre productos
        - perfil_productos: historial completo de transacciones del entrenamiento
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        art = ctx.model["artefactos"]
        ctx.extra["model_knn"] = art["model_knn"]
        ctx.extra["model_xgb_cantidad"] = art["model_xgb_cantidad"]
        ctx.extra["model_apriori"] = art["model_apriori"]
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

        cliente_id = ctx.parameters.get("cliente_id")
        if cliente_id is None:
            ctx.errors.append("ValidateClienteStep: 'cliente_id' no proporcionado.")
            return ctx

        customer_profile = ctx.extra["model_knn"]["data"]
        if cliente_id not in customer_profile["cliente_id"].values:
            ctx.errors.append(
                f"ValidateClienteStep: cliente {cliente_id} no tiene historial."
            )
            return ctx

        logger.info("cliente_validado", cliente_id=cliente_id)
        return ctx


class KnnFindNeighborsStep(BaseStep[PredictContext]):
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
        cliente_id = ctx.parameters.get("cliente_id")

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
        logger.info("knn_vecinos_encontrados", cliente_id=cliente_id, segmento=segmento, n_vecinos=len(vecinos_ids))
        return ctx


class KnnBuildCandidatesStep(BaseStep[PredictContext]):
    """
    Construye la lista de productos candidatos a recomendar a partir del
    historial de compras de los vecinos.

    Construye una matriz pivot (cliente x producto) desde perfil_productos
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

        solo_nuevos = ctx.parameters.get("solo_nuevos", True)
        if solo_nuevos:
            cliente_id = ctx.parameters.get("cliente_id")
            productos_propios = set(
                perfil_productos[perfil_productos["cliente_id"] == cliente_id][
                    "nombre_producto"
                ].unique()
            )
            candidatos = [p for p in candidatos_vecinos if p not in productos_propios]
            logger.info("knn_candidatos", total_vecinos=len(candidatos_vecinos), nuevos_para_cliente=len(candidatos))
        else:
            candidatos = candidatos_vecinos
            logger.info("knn_candidatos", total=len(candidatos), solo_nuevos=False)

        ctx.extra["candidatos"] = candidatos
        df = build_features_candidatos(
            candidatos=candidatos,
            cliente_id=ctx.parameters.get("cliente_id"),
            perfil_productos=ctx.extra["perfil_productos"],
            segmento=ctx.extra["segmento"],
            fuente_nueva="vecinos",
        )
        ctx.extra["pct_vecinos"] = pct_vecinos
        ctx.extra["df_features_knn"] = df
        return ctx


class KnnRankAndPredictStep(BaseStep[PredictContext]):
    """
    Genera las recomendaciones KNN+XGB combinando el score de vecinos con
    la predicción de cantidad del XGBRegressor.

    Parámetros configurables vía ctx.extra:
        - top_n (default 10): número máximo de productos a retornar
        - cantidad_minima (default 1.0): filtra productos con cantidad
          sugerida menor a este valor

    Flujo:
        1. Asigna score = pct_vecinos (fracción de vecinos que compran el producto).
        2. Pre-selecciona top Nx3 por score para reducir el costo de inferencia.
        3. Predice la cantidad sugerida con XGBRegressor.
        4. Filtra por cantidad_minima y retorna top N ordenados por score.

    Guarda en ctx.extra["recomendaciones_knn_xgb"]: DataFrame con
    nombre_producto, cantidad_sugerida, score, fuente.
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        top_n = ctx.parameters.get("top_n", TOP_N)
        cantidad_minima = ctx.parameters.get("cantidad_minima", CANTIDAD_MINIMA)

        model_xgb_cantidad = ctx.extra["model_xgb_cantidad"]
        pct_vecinos = ctx.extra["pct_vecinos"]
        features = model_xgb_cantidad["features"]

        df = ctx.extra["df_features_knn"].copy()
        df["score"] = df["nombre_producto"].map(pct_vecinos).fillna(0)

        top_df = df.nlargest(top_n * 3, "score").copy()

        df_cantidad = top_df.copy()
        df_cantidad[CAT_FEATURES] = model_xgb_cantidad["encoder"].transform(
            df_cantidad[CAT_FEATURES].fillna("DESCONOCIDO")
        )
        top_df["cantidad_sugerida"] = np.maximum(
            model_xgb_cantidad["model"].predict(df_cantidad[features]), 0
        ).round(2)

        resultado = top_df[top_df["cantidad_sugerida"] >= cantidad_minima]
        resultado = resultado.nlargest(top_n, "score")

        recomendaciones = resultado[
            ["nombre_producto", "cantidad_sugerida", "score", "_fuente"]
        ].rename(columns={"_fuente": "fuente"})
        ctx.extra["recomendaciones_knn_xgb"] = recomendaciones
        logger.info("knn_recomendaciones_generadas", n_resultados=len(recomendaciones), top_n=top_n)
        return ctx


class AprioriBuildCandidatesStep(BaseStep[PredictContext]):
    """
    Construye candidatos a recomendar usando las reglas de asociación Apriori.

    Busca reglas cuyo antecedente coincida con algún producto que el cliente
    ya compra. Los consequents son los productos candidatos. El score de cada
    candidato es el máximo de confidence x lift entre todas las reglas que lo
    generan.

    Si solo_nuevos=True (default), excluye los productos que el cliente ya compra.

    Guarda en ctx.extra:
        - candidatos_apriori: lista de productos candidatos
        - scores_apriori: Serie indexada por nombre_producto con el score
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        rules = ctx.extra["model_apriori"]["rules"]
        perfil_productos = ctx.extra["perfil_productos"]
        cliente_id = ctx.parameters.get("cliente_id")

        productos_cliente = set(
            perfil_productos[perfil_productos["cliente_id"] == cliente_id][
                "nombre_producto"
            ].unique()
        )

        reglas_match = rules[rules["antecedent"].isin(productos_cliente)].copy()

        if reglas_match.empty:
            logger.warning("apriori_sin_reglas", cliente_id=cliente_id, productos_cliente=len(productos_cliente))
            ctx.extra["candidatos_apriori"] = []
            ctx.extra["scores_apriori"] = pd.Series(dtype=float)
            ctx.extra["df_features_apriori"] = pd.DataFrame()
            return ctx

        reglas_match["score"] = reglas_match["confidence"] * reglas_match["lift"]

        # Para cada consecuente, tomar la regla con mayor score y conservar su antecedente
        idx_mejor = reglas_match.groupby("consequent")["score"].idxmax()
        mejores_reglas = reglas_match.loc[idx_mejor].set_index("consequent")
        scores = mejores_reglas["score"]          # mismo orden de índice que antecedente_map
        antecedente_map = mejores_reglas["antecedent"]

        solo_nuevos = ctx.parameters.get("solo_nuevos", True)
        if solo_nuevos:
            mask = ~scores.index.isin(productos_cliente)
            scores = scores[mask]
            antecedente_map = antecedente_map[mask]

        scores = scores.sort_values(ascending=False)  # ordenar DESPUÉS del filtro

        df = build_features_candidatos(
            candidatos=scores.index.tolist(),
            cliente_id=ctx.parameters.get("cliente_id"),
            perfil_productos=ctx.extra["perfil_productos"],
            segmento=ctx.extra["segmento"],
            fuente_nueva="apriori",
        )

        ctx.extra["df_features_apriori"] = df
        ctx.extra["candidatos_apriori"] = scores.index.tolist()
        ctx.extra["scores_apriori"] = scores
        ctx.extra["antecedente_apriori"] = antecedente_map
        logger.info("apriori_candidatos", n_candidatos=len(scores), cliente_id=cliente_id)
        return ctx


class AprioriRankAndPredictStep(BaseStep[PredictContext]):
    """
    Genera las recomendaciones Apriori+XGB combinando el score de las reglas
    (confidence x lift) con la predicción de cantidad del XGBRegressor.

    Parámetros configurables vía ctx.parameters:
        - top_n (default 10): número máximo de productos a retornar
        - cantidad_minima (default 1.0): filtra productos con cantidad menor

    Guarda en ctx.extra["recomendaciones_apriori_xgb"]: DataFrame con
    nombre_producto, cantidad_sugerida, score, fuente.
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        top_n = ctx.parameters.get("top_n", TOP_N)
        cantidad_minima = ctx.parameters.get("cantidad_minima", CANTIDAD_MINIMA)

        df = ctx.extra["df_features_apriori"].copy()

        if df.empty:
            ctx.extra["recomendaciones_apriori_xgb"] = pd.DataFrame(
                columns=["nombre_producto", "antecedente", "cantidad_sugerida", "score", "fuente"]
            )
            return ctx

        model_xgb_cantidad = ctx.extra["model_xgb_cantidad"]
        scores_apriori = ctx.extra["scores_apriori"]
        antecedente_map = ctx.extra["antecedente_apriori"]
        features = model_xgb_cantidad["features"]

        df["score"] = df["nombre_producto"].map(scores_apriori).fillna(0)
        df["antecedente"] = df["nombre_producto"].map(antecedente_map)

        top_df = df.nlargest(top_n * 3, "score").copy()

        df_cantidad = top_df.copy()
        df_cantidad[CAT_FEATURES] = model_xgb_cantidad["encoder"].transform(
            df_cantidad[CAT_FEATURES].fillna("DESCONOCIDO")
        )
        top_df["cantidad_sugerida"] = np.maximum(
            model_xgb_cantidad["model"].predict(df_cantidad[features]), 0
        ).round(2)

        resultado = top_df[top_df["cantidad_sugerida"] >= cantidad_minima]
        resultado = resultado.nlargest(top_n, "score")

        recomendaciones = resultado[
            ["nombre_producto", "antecedente", "cantidad_sugerida", "score", "_fuente"]
        ].rename(columns={"_fuente": "fuente"})
        ctx.extra["recomendaciones_apriori_xgb"] = recomendaciones
        logger.info("apriori_recomendaciones_generadas", n_resultados=len(recomendaciones), top_n=top_n)
        return ctx


class BuildResponseStep(BaseStep[PredictContext]):
    """
    Ensambla la respuesta final combinando ambas fuentes de recomendación
    en un dict con dos claves:
        - knn_xgb: recomendaciones basadas en vecinos similares + XGBoost
        - apriori_xgb: recomendaciones basadas en reglas de asociación + XGBoost
    """

    def execute(self, ctx: PredictContext) -> PredictContext:
        knn = ctx.extra["recomendaciones_knn_xgb"]
        apriori = ctx.extra["recomendaciones_apriori_xgb"]
        ctx.data_response = {
            "knn_xgb": knn.to_dict(orient="records"),
            "apriori_xgb": apriori.to_dict(orient="records"),
        }
        logger.info(
            "respuesta_ensamblada",
            cliente_id=ctx.parameters.get("cliente_id"),
            n_knn=len(knn),
            n_apriori=len(apriori),
        )
        return ctx
