from typing import Any, cast
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from app.domain.ml.base_context import PredictContext
from app.domain.ml.base_step import BaseTrainingStep

class PredictStep(BaseTrainingStep[PredictContext]):
    def execute(self, ctx: PredictContext) -> PredictContext:

        scaled= ctx.extra.get("x_scaled")
        if scaled is None:
            ctx.errors.append("PredictStep: x_scaled es None")
        x_scaled: NDArray[Any] = cast(NDArray[Any], scaled)
        
        df = ctx.extra.get("perfil_clientes")
        if df is None:
            ctx.errors.append("PredictStep: perfil_clientes es None")
        perfil_clientes: pd.DataFrame = cast(pd.DataFrame, df)
            
        id_cliente = ctx.data.get("id_cliente")
        idx_cliente = perfil_clientes[perfil_clientes['ID_Cliente'] == id_cliente].index[0]
        
        model = ctx.model["model"]
        df = ctx.model["df"]
        
        distancias, indices = model.kneighbors(x_scaled[idx_cliente].reshape(1, -1))
        vecinos_ids = perfil_clientes.iloc[indices[0]]['ID_Cliente'].tolist()
        
        sugerencia_productos = df[df['ID_Cliente'].isin(vecinos_ids)].groupby('Producto').agg({
            'CantidadVendida': 'mean'
        }).sort_values(ascending=False, by='CantidadVendida').head(5)
        
        ctx.data_response = sugerencia_productos
        return ctx
    