import pandas as pd

from app.domain.ml.predict_params import BuildFeaturesRequest, ParetoConfig


def apply_pareto(df: pd.DataFrame, config: ParetoConfig) -> pd.DataFrame:
    """
    Filtra un DataFrame de recomendaciones aplicando la ley de Pareto sobre
    cantidad_sugerida.

    Pasos:
        1. Excluye productos con cantidad_sugerida < cantidad_minima.
        2. Ordena por cantidad_sugerida descendente.
        3. Retiene los productos que cubren el PORCENTAJE_PARETO (20%) del
           volumen total acumulado — incluyendo el producto que cruza el umbral.
        4. Aplica top_n como techo máximo.

    Args:
        df: DataFrame con al menos las columnas 'cantidad_sugerida'.
        config: ParetoConfig con top_n, cantidad_minima y porcentaje_volumen.

    Returns:
        DataFrame filtrado, ordenado por cantidad_sugerida descendente.
    """
    df = df[df["cantidad_sugerida"] >= config.cantidad_minima].copy()
    if df.empty:
        return df
    df = df.sort_values("cantidad_sugerida", ascending=False)
    total = df["cantidad_sugerida"].sum()
    cum_pct = df["cantidad_sugerida"].cumsum() / total
    mask = cum_pct.shift(fill_value=0) < config.porcentaje_volumen
    return df[mask].head(config.top_n)


def build_features_candidatos(req: BuildFeaturesRequest) -> pd.DataFrame:
    """
    Construye el DataFrame de features para una lista de productos candidatos,
    en el formato que espera XGBRegressor.

    Para cada producto distingue dos casos:
        - historial_propio: el cliente ya compró el producto. Se usan sus propios
          promedios, recencia y datos de la última compra.
        - fuente_nueva: el cliente nunca compró el producto. Se usan los promedios
          globales del producto y los datos base del cliente como contexto.

    Args:
        req: BuildFeaturesRequest con candidatos, cliente_id, perfil_productos,
             segmento y fuente_nueva.

    Returns:
        DataFrame con una fila por candidato y todas las features requeridas.
    """
    fecha_max = pd.to_datetime(req.perfil_productos["fecha_venta"]).max()
    mes_actual = fecha_max.month

    historial_cliente = req.perfil_productos[req.perfil_productos["cliente_id"] == req.cliente_id]
    if historial_cliente.empty:
        return pd.DataFrame()
    ctx_base = historial_cliente.sort_values("fecha_venta").iloc[-1]

    filas = []
    for producto in req.candidatos:
        hist_prod = historial_cliente[historial_cliente["nombre_producto"] == producto]
        prod_info = req.perfil_productos[req.perfil_productos["nombre_producto"] == producto]

        if len(hist_prod) > 0:
            ultima = hist_prod.sort_values("fecha_venta").iloc[-1]
            promedio_historico = hist_prod["cantidad_vendida"].mean()
            promedio_ultimas_3 = hist_prod["cantidad_vendida"].tail(3).mean()
            dias_entre_compras = hist_prod["dias_entre_compras"].mean()
            dias_desde_ultima_compra = (fecha_max - ultima["fecha_venta"]).days
            marca = ultima["marca"]
            linea_producto = ultima["linea_producto"]
            fuente = "historial_propio"
            num_productos_distintos = int(hist_prod["nombre_producto"].nunique())
            importe_total_cliente = float(hist_prod["cantidad_vendida"].sum())
            frecuencia_promedio_cliente = float(hist_prod["dias_entre_compras"].mean())
            cantidad_productos_comprados = int(hist_prod["nombre_producto"].count())
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
                prod_info["marca"].iloc[0] if len(prod_info) > 0 else ctx_base["marca"]
            )
            linea_producto = (
                prod_info["linea_producto"].iloc[0]
                if len(prod_info) > 0
                else ctx_base["linea_producto"]
            )
            fuente = req.fuente_nueva
            num_productos_distintos = int(prod_info["nombre_producto"].nunique()) if len(prod_info) > 0 else 0
            importe_total_cliente = float(prod_info["cantidad_vendida"].sum()) if len(prod_info) > 0 else 0.0
            frecuencia_promedio_cliente = float(prod_info["dias_entre_compras"].mean()) if len(prod_info) > 0 else 0.0
            cantidad_productos_comprados = 0

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
                "segmento": req.segmento,
                "num_productos_distintos": num_productos_distintos,
                "importe_total_cliente": importe_total_cliente,
                "frecuencia_promedio_cliente": frecuencia_promedio_cliente,
                "cantidad_productos_comprados": cantidad_productos_comprados,
                "_fuente": fuente,
            }
        )

    return pd.DataFrame(filas)
