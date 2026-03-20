from datetime import datetime
from typing import Any, cast
import pandas as pd
from app.domain.ml.base_context import PredictContext
from app.domain.ml.base_step import BaseTrainingStep

class PredictStep(BaseTrainingStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:
        df = ctx.extra.get("perfil_clientes")
        if df is None:
            ctx.errors.append("PredictStep: perfil_clientes es None, StandarizationStep no se ejecutó.")
            return ctx
        perfil_clientes: pd.DataFrame = cast(pd.DataFrame, df)
            
        id_cliente = ctx.data.get("id_cliente")
        perfil = perfil_clientes[perfil_clientes['ID_Cliente'] == id_cliente].iloc[0]
        print(f"Perfil del cliente {id_cliente}:")
        print(perfil)
        print(type(perfil))
        
        model = ctx.model["model"]
        df = cast(pd.DataFrame, ctx.model["df"])
        print("DataFrame para predicción:")
        print(df.head())
        
        datos_prediccion = {
            "ID_Cliente": id_cliente,
            "Producto": "",
            "CantidadVendida":0,
            "LineaProducto": "",
            "Marca": "",
            "ClasificacionCliente": "perfil oficina",
            "ID_Ruta" :0,
            "ID_Zona":0,
            "Sucursal": "",
            "DiasEntreCompras":0,
            "PromedioHistorico" :0,
            "DiasDesdeUltimaCompra":0,
            "DiaSemana": datetime.now().weekday(),
            "Mes": datetime.now().month,
            "Cluster": perfil["Cluster"].item(),
        }
        datos_prediccion = pd.DataFrame([datos_prediccion])
        print("Datos de predicción antes de la transformación:")
        print(datos_prediccion)

        datos_prediccion =   model["prep"].transform(datos_prediccion)
        distancias, indices = model["knn"].kneighbors(datos_prediccion)
        print("Distancias a los vecinos más cercanos:", distancias)
        print("Índices de los vecinos más cercanos:", indices)
        sugerencia_productos = df.iloc[indices[0]].copy()
        sugerencia_productos["distancia"] = distancias[0]
        sugerencia_productos = (
            sugerencia_productos
            .sort_values('distancia', ascending=False)
            .head(10)
        )
        # vecinos_ids = df.iloc[indices[0]]['ID_Cliente'].tolist()
        # print("Vecinos más cercanos (IDs):", vecinos_ids)

        # sugerencia_productos = df[df['ID_Cliente'].isin(vecinos_ids)].groupby('Producto').agg({
        #     'Producto': 'first',
        #     'CantidadVendida': 'mean'

        # }).sort_values(ascending=False, by='CantidadVendida').head(5)
        
        ctx.data_response = sugerencia_productos
        print("Sugerencia de productos:")
        print(sugerencia_productos)
        return ctx
    