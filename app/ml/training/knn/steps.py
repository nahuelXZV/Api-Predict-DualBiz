import joblib
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.domain.ml.base_model import BaseMLModel
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
    
    
class AddDerivedFeatureStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.raw_data is None:
            ctx.errors.append("CleanDataStep: raw_data es None, LoadDataStep no se ejecutó.")
            return ctx
        
        df = ctx.raw_data.copy()   
        df['FechaVenta'] = pd.to_datetime(df['FechaVenta'])
        df = df.sort_values(by=['ID_Cliente', 'Producto', 'FechaVenta'])
        
        # 4. Cálculo de Features (Frecuencia, Promedios, Recencia)
        df['FechaAnterior'] = df.groupby(['ID_Cliente', 'Producto'])['FechaVenta'].shift(1)
        df['DiasEntreCompras'] = (df['FechaVenta'] - df['FechaAnterior']).dt.days
        df['PromedioHistorico'] = df.groupby(['ID_Cliente', 'Producto'])['CantidadVendida'].transform('mean')

        fecha_maxima = df['FechaVenta'].max()
        ultima_compra = df.groupby(['ID_Cliente', 'Producto'])['FechaVenta'].transform('max')
        df['DiasDesdeUltimaCompra'] = (fecha_maxima - ultima_compra).dt.days

        # 5. Estacionalidad
        df['DiaSemana'] = df['FechaVenta'].dt.dayofweek
        df['Mes'] = df['FechaVenta'].dt.month

        # 6. Limpieza y Exportación
        df['DiasEntreCompras'] = df['DiasEntreCompras'].fillna(0)

        ctx.raw_data = df
        print("Dataset con features derivados:")
        print(df.head())
        return ctx
    
        
class CleanColumnsStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.raw_data is None:
            ctx.errors.append("CleanDataStep: raw_data es None, CleanColumnsStep no se ejecutó.")
            return ctx
        
        df = ctx.raw_data.copy()   
        df_limpio = df.dropna(subset=['ID_Cliente', 'Producto', 'CantidadVendida'])
        # df_limpio = df_limpio.dropna()
        df_limpio = df_limpio[df_limpio['CantidadVendida'] > 0]
        df_limpio = df_limpio.drop(
            columns=[
                # "Producto",
                "Nombre_Ruta",
                "Nombre_Zona",
                "FechaAnterior",
                "FechaVenta",
                "Vendedor",
            ]
        )
        ctx.clean_data = df_limpio
        print("Dataset limpio:")
        print(df_limpio.info())
        print(df_limpio.head())
        return ctx
    
class SegmentCustomersStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("CleanDataStep: clean_data es None, TransformColumnsStep no se ejecutó.")
            return ctx
        
        df = ctx.clean_data.copy()   
        perfil_clientes = df.groupby('ID_Cliente').agg({
                'CantidadVendida': 'sum',      # Volumen total
                'PromedioHistorico': 'mean',   # Tamaño de pedido habitual
                'DiasEntreCompras': 'mean',    # Frecuencia de visita
                'Mes': 'nunique'               # Constancia a lo largo del año
            }).reset_index().fillna(0)
      
        scaler = StandardScaler()
        features_to_scale = ['CantidadVendida', 'PromedioHistorico', 'DiasEntreCompras', 'Mes']
        X_scaled = scaler.fit_transform(perfil_clientes[features_to_scale])
        
        kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, n_init=10)
        segmento = kmeans.fit_predict(X_scaled)
        perfil_clientes['Cluster'] = segmento
        df = df.merge(
            perfil_clientes[['ID_Cliente', 'Cluster']],
            on='ID_Cliente',
            how='left'
        )
        
        ctx.extra["x_scaled"] = X_scaled
        ctx.extra["perfil_clientes"] = perfil_clientes
        ctx.clean_data = df
        
        print("Datos Segmentados:")
        print(perfil_clientes.head())
        print(df.head())
        print(df.info())
        return ctx
    

    
class SaveModelStep(BaseTrainingStep[TrainingContext]): 
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("CleanDataStep: clean_data es None, StandarizationStep no se ejecutó.")
            return ctx
        
        cat_features = [
            "LineaProducto",
            "Marca",
            "ClasificacionCliente",
            "LineaProducto",
            "Sucursal",
        ]

        num_features = [
            "ID_Ruta",
            "ID_Zona",
            "CantidadVendida",
            "PromedioHistorico",
            "DiasEntreCompras",
            "DiasDesdeUltimaCompra",
            "Mes",
            "Cluster"
        ]
        
        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
                ("num", StandardScaler(), num_features)
            ]
        )

        # df = ctx.extra["x_scaled"].copy()   
        df = ctx.clean_data.copy()   
        model = Pipeline([
            ("prep", preprocess),
            ("knn", NearestNeighbors(
                n_neighbors=29,
                metric="cosine"
            ))
        ])
        path_model = MODEL_PATH_BASE / f'modelo_{ctx.model_name}_{ctx.version}.pkl'
        ctx.extra["path_model"] = path_model

        model.fit(df)
        joblib.dump(
            {
                "model": model,
                "df": ctx.clean_data
            },
            str(path_model)
        )
        print(f"Modelo guardado en {path_model}")
        return ctx
    
    
class RegistryModelStep(BaseTrainingStep[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        path_model = str(ctx.extra.get("path_model"))
   
        meta_data = ModelMetadata(name=ctx.model_name, version=ctx.version, path_model=path_model,extra=ctx.extra) #sacar el extra de aqui y guardar en el archivo pkl
        model = KnnModel(metadata=meta_data)
        model.load(path_model)
        
        model_registry.register(
            name = ctx.model_name,
            model = model,
        )
        return ctx