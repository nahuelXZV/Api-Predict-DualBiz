import joblib
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from app.domain.base_step import BaseTrainingStep
from app.ml.training.context import TrainingContext

BASE_DIR = Path(__file__).resolve().parents[0]  
DATA_PATH = BASE_DIR / "storage" / "data" / "base.csv"
MODEL_PATH = BASE_DIR / "storage" / "models" / "modelo_knn.pkl"

class LoadDataStep(BaseTrainingStep):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        df = pd.read_csv(DATA_PATH)
        print("Dataset original")
        print(df.head())
        df.info()
        
        ctx.raw_data = df
        ctx.extra["cod_cliente"] = 14111
        return ctx
    
    
class AddDerivedFeatureStep(BaseTrainingStep):

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

        # Promedio móvil y Recencia
        df['PromedioUltimas3'] = (df.groupby(['ID_Cliente', 'Producto'])['CantidadVendida']
                                .rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True))

        fecha_maxima = df['FechaVenta'].max()
        ultima_compra = df.groupby(['ID_Cliente', 'Producto'])['FechaVenta'].transform('max')
        df['DiasDesdeUltimaCompra'] = (fecha_maxima - ultima_compra).dt.days

        # 5. Estacionalidad
        df['DiaSemana'] = df['FechaVenta'].dt.dayofweek
        df['Mes'] = df['FechaVenta'].dt.month

        # 6. Limpieza y Exportación
        df['DiasEntreCompras'] = df['DiasEntreCompras'].fillna(0)
        
        ctx.clean_data = df
        return ctx
    
        
class CleanColumnsStep(BaseTrainingStep):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("CleanDataStep: clean_data es None, CleanColumnsStep no se ejecutó.")
            return ctx
        
        df = ctx.clean_data.copy()   
        df_limpio = df.dropna(subset=['ID_Cliente', 'Producto', 'CantidadVendida'])
        df_limpio = df_limpio[df_limpio['CantidadVendida'] > 0]
        
        ctx.clean_data = df
        return ctx
    
class TransformColumnsStep(BaseTrainingStep):
    
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("CleanDataStep: clean_data es None, TransformColumnsStep no se ejecutó.")
            return ctx
        
        df = ctx.clean_data.copy()   
        df['FechaVenta'] = pd.to_datetime(df['FechaVenta'], errors='coerce')
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
        perfil_clientes['Cluster'] = kmeans.fit_predict(X_scaled)
        
        ctx.extra["X_scaled"] = X_scaled
        ctx.extra["perfil_clientes"] = perfil_clientes
        ctx.clean_data = df
        return ctx
    

    
class SaveModelStep(BaseTrainingStep): 
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        if ctx.clean_data is None:
            ctx.errors.append("CleanDataStep: clean_data es None, StandarizationStep no se ejecutó.")
            return ctx
        
        df = ctx.clean_data.copy()   
        model = Pipeline([
            ("knn", NearestNeighbors(
                n_neighbors=29,
                metric="cosine"
            ))
        ])

        model.fit(df)
        joblib.dump(
            {
                "model": model,
                "df": df
            },
            MODEL_PATH
        )
        return ctx
    
    
