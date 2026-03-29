import pandas as pd


<<<<<<< HEAD
def apply_pareto(
    df: pd.DataFrame, top_n: int, cantidad_minima: float, porcentaje_volumen: float
) -> pd.DataFrame:
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
        top_n: Número máximo de productos a retornar.
        cantidad_minima: Umbral mínimo de cantidad_sugerida para incluir un producto.
        porcentaje_volumen: Porcentaje del volumen total de cantidad_sugerida que se quiere cubrir.

    Returns:
        DataFrame filtrado, ordenado por cantidad_sugerida descendente.
    """
    df = df[df["cantidad_sugerida"] >= cantidad_minima].copy()
    if df.empty:
        return df
    df = df.sort_values("cantidad_sugerida", ascending=False)
    total = df["cantidad_sugerida"].sum()
    cum_pct = df["cantidad_sugerida"].cumsum() / total
    mask = cum_pct.shift(fill_value=0) < porcentaje_volumen
    return df[mask].head(top_n)


=======
>>>>>>> bfaa3fb2cac645cc53a2b75ba8d9a7a20814fa99
def build_features_candidatos(
    candidatos: list,
    cliente_id,
    perfil_productos: pd.DataFrame,
    segmento: int,
    fuente_nueva: str = "vecinos",
) -> pd.DataFrame:
    """
    Construye el DataFrame de features para una lista de productos candidatos,
    en el formato que espera XGBRegressor.

    Para cada producto distingue dos casos:
        - historial_propio: el cliente ya compró el producto. Se usan sus propios
          promedios, recencia y datos de la última compra.
        - fuente_nueva: el cliente nunca compró el producto. Se usan los promedios
          globales del producto y los datos base del cliente como contexto.

    Args:
        candidatos: Lista de nombres de producto a evaluar.
        cliente_id: Identificador del cliente.
        perfil_productos: DataFrame completo de transacciones del entrenamiento.
        segmento: Segmento KMeans del cliente (feature para XGBRegressor).
        fuente_nueva: Etiqueta de fuente para productos sin historial propio.
                      "vecinos" para KNN, "apriori" para Apriori.

    Returns:
        DataFrame con una fila por candidato y todas las features requeridas.
    """
    fecha_max = pd.to_datetime(perfil_productos["fecha_venta"]).max()
    mes_actual = fecha_max.month

    historial_cliente = perfil_productos[perfil_productos["cliente_id"] == cliente_id]
    ctx_base = historial_cliente.sort_values("fecha_venta").iloc[-1]

    filas = []
    for producto in candidatos:
        hist_prod = historial_cliente[historial_cliente["nombre_producto"] == producto]
        prod_info = perfil_productos[perfil_productos["nombre_producto"] == producto]

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
<<<<<<< HEAD
                prod_info["marca"].iloc[0] if len(prod_info) > 0 else ctx_base["marca"]
=======
                prod_info["marca"].iloc[0]
                if len(prod_info) > 0
                else ctx_base["marca"]
>>>>>>> bfaa3fb2cac645cc53a2b75ba8d9a7a20814fa99
            )
            linea_producto = (
                prod_info["linea_producto"].iloc[0]
                if len(prod_info) > 0
                else ctx_base["linea_producto"]
            )
            fuente = fuente_nueva

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

    return pd.DataFrame(filas)
