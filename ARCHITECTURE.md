# Arquitectura del Proyecto — Api-Predict-DualBiz

API REST para entrenamiento y predicción de modelos de Machine Learning. Construida con Django REST Framework siguiendo los principios de **Clean Architecture** y **SOLID**.

---

## Tabla de contenido

1. [Visión general](#1-visión-general)
2. [Estructura de carpetas](#2-estructura-de-carpetas)
3. [Capas de la arquitectura](#3-capas-de-la-arquitectura)
4. [Flujo de un request](#4-flujo-de-un-request)
5. [El sistema de Pipeline](#5-el-sistema-de-pipeline)
6. [Patrones de diseño](#6-patrones-de-diseño)
7. [Modelo pedido_sugerido](#7-modelo-pedido_sugerido)
8. [DataSources](#8-datasources)
9. [Registro de modelos](#9-registro-de-modelos)
10. [Registro de pipelines](#10-registro-de-pipelines)
11. [Configuración y settings](#11-configuración-y-settings)
12. [Logging](#12-logging)
13. [Endpoints de la API](#13-endpoints-de-la-api)
14. [Cómo agregar un nuevo modelo](#14-cómo-agregar-un-nuevo-modelo)

---

## 1. Visión general

El proyecto está organizado en **cuatro capas** con dependencias unidireccionales. Ninguna capa conoce a las que están por encima de ella.

```
┌─────────────────────────────────────────┐
│           PRESENTATION                  │  Django REST Framework
│   (Endpoints, Serializers, Views)       │
└───────────────────┬─────────────────────┘
                    │ usa
┌───────────────────▼─────────────────────┐
│           APPLICATION                   │  Casos de uso
│   (PredictService, TrainingService)     │
└───────────────────┬─────────────────────┘
                    │ usa
┌───────────────────▼─────────────────────┐
│          INFRASTRUCTURE                 │  Implementaciones concretas
│  (Pipelines, Models, DataSources, ML)   │
└───────────────────┬─────────────────────┘
                    │ implementa
┌───────────────────▼─────────────────────┐
│             DOMAIN                      │  Sin dependencias externas
│  (Abstracciones, DTOs, Excepciones)     │
└─────────────────────────────────────────┘
```

**Regla fundamental:** las capas superiores pueden conocer a las inferiores, nunca al revés. El `domain` no importa nada de `infrastructure` ni de `presentation`.

---

## 2. Estructura de carpetas

```
Api-Predict-DualBiz/
│
├── config/                          # Configuración Django
│   ├── settings.py                  # INSTALLED_APPS, REST_FRAMEWORK, etc.
│   ├── urls.py                      # Rutas raíz (API + web + admin + Swagger)
│   ├── asgi.py
│   └── wsgi.py
│
├── app/
│   │
│   ├── domain/                      # ① CAPA DE DOMINIO
│   │   ├── core/
│   │   │   ├── config.py            # Settings con pydantic-settings
│   │   │   ├── exceptions.py        # Excepciones tipadas del dominio
│   │   │   └── logging.py           # Setup de structlog
│   │   ├── dtos/
│   │   │   ├── predict_dto.py       # PredictRequestDTO, PredictResponseDTO
│   │   │   ├── training_dto.py      # TrainRequestDTO, TrainResponseDTO
│   │   │   └── response_dto.py      # ResponseDTO[T], ResponseEnvelope
│   │   └── ml/
│   │       ├── abstractions/
│   │       │   ├── ml_model_abc.py  # Contrato de todo modelo ML
│   │       │   ├── step_abc.py      # Contrato de todo paso de pipeline
│   │       │   ├── pipeline_base.py # Orquestador genérico de pasos
│   │       │   └── data_source_abc.py  # Contrato de toda fuente de datos
│   │       ├── model_metadata.py    # Dataclass con info del modelo
│   │       ├── model_registry.py    # Registro thread-safe de modelos
│   │       ├── pipeline_context.py  # BaseContext, TrainingContext, PredictContext
│   │       ├── predict_params.py    # ParetoConfig, BuildFeaturesRequest
│   │       └── training_params.py   # SearchCVConfig
│   │
│   ├── application/                 # ② CAPA DE APLICACIÓN
│   │   └── services/
│   │       ├── predict_service.py
│   │       ├── training_service.py
│   │       └── model_manager_service.py
│   │
│   ├── infrastructure/              # ③ CAPA DE INFRAESTRUCTURA
│   │   └── ml/
│   │       ├── pipeline_registry.py     # @register_pipeline + get_pipeline()
│   │       ├── model_manager.py         # Orquesta train y predict
│   │       ├── load_models.py           # Carga modelos al iniciar la app
│   │       ├── data_sources/
│   │       │   ├── data_source_factory.py
│   │       │   ├── csv_data_source_strategy.py
│   │       │   └── sqlserver_data_source_strategy.py
│   │       ├── models/
│   │       │   └── pedido_sugerido_model.py
│   │       ├── training/
│   │       │   └── pedido_sugerido/
│   │       │       ├── pipeline.py      # PedidoSugeridoPipeline
│   │       │       ├── steps.py         # 10 pasos de entrenamiento
│   │       │       ├── queries.py       # Query SQL por defecto
│   │       │       └── utils.py
│   │       └── predict/
│   │           └── pedido_sugerido/
│   │               ├── pipeline.py      # predict_pedido_sugerido_pipeline()
│   │               ├── steps.py         # 10 pasos de predicción
│   │               └── utils.py
│   │
│   ├── presentation/                # ④ CAPA DE PRESENTACIÓN
│   │   ├── api/
│   │   │   ├── responses.py             # success_response(), error_response()
│   │   │   ├── exception_handler.py     # Handler global de errores DRF
│   │   │   └── v1/
│   │   │       ├── urls.py
│   │   │       └── endpoints/
│   │   │           ├── health.py
│   │   │           ├── predict.py
│   │   │           ├── training.py
│   │   │           ├── models.py
│   │   │           └── serializers/
│   │   │               ├── predict/     # request + response serializers
│   │   │               ├── train/       # request + response serializers
│   │   │               └── model/       # metadata serializer
│   │   └── web/
│   │       ├── urls.py
│   │       └── views/
│   │
│   └── apps.py                      # AppConfig.ready() → setup inicial
│
├── storage/
│   ├── models/                      # Modelos .pkl entrenados
│   └── data/                        # Archivos CSV para entrenamiento local
│
├── logs/                            # Logs estructurados (JSON)
├── requirements.txt
├── .env.example
└── manage.py
```

---

## 3. Capas de la arquitectura

### ① Domain — El núcleo

Sin dependencias externas de ningún framework. Solo Python puro, `abc`, `dataclasses`, y `pandas`/`numpy` en los parámetros ML.

**¿Por qué?** Si mañana migramos de Django a FastAPI, o de SQL Server a MongoDB, el dominio no cambia.

Contiene:
- **Abstracciones** (ABCs): contratos que el resto del sistema debe cumplir
- **DTOs**: objetos de transferencia de datos entre capas
- **Excepciones**: tipadas y descriptivas (`ModelNotFoundError`, `ModelNotReadyError`, etc.)
- **ModelRegistry**: registro thread-safe de modelos cargados en memoria
- **Contextos de pipeline**: `TrainingContext` y `PredictContext`

### ② Application — Casos de uso

Orquesta el flujo entre presentación e infraestructura. No contiene lógica de negocio, solo coordinación.

```python
# training_service.py
class TrainingService:
    def run(self, request: TrainRequestDTO) -> TrainResponseDTO:
        return model_manager.train(request)
```

**¿Por qué existe esta capa si es tan delgada?** Porque es el punto de entrada limpio para cualquier cambio de orquestación. Si necesitamos agregar autorización, auditoría o notificaciones antes/después de entrenar, se agrega aquí sin tocar ni la presentación ni la infraestructura.

### ③ Infrastructure — Implementaciones concretas

Implementa todas las abstracciones del dominio:

| Abstracción (domain) | Implementación (infrastructure) |
|---|---|
| `DataSourceABC` | `CsvDataSourceStrategy`, `SqlServerDataSourceStrategy` |
| `MLModelABC` | `PedidoSugeridoModel` |
| `StepABC` | `LoadDataStep`, `EdaCleanDataStep`, ... (20+ steps) |
| `PipelineBase` | `PedidoSugeridoPipeline` |

### ④ Presentation — API REST

Valida requests, serializa responses, maneja errores HTTP. No contiene lógica de negocio.

```python
# predict.py
def post(self, request):
    serializer = PredictRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)          # valida y parsea
    data = cast(dict, serializer.validated_data)
    result = self.service.predict(...)                  # delega al service
    return success_response(data=result, message="...") # serializa response
```

---

## 4. Flujo de un request

### Predicción

```
POST /api/v1/predict/
{
  "model_name": "pedido_sugerido",
  "parameters": {
    "cliente_id": 14111,
    "solo_nuevos": true,
    "top_n": 50,
    "cantidad_minima": 1.0,
    "porcentaje_pareto": 20
  }
}
```

```
PredictView.post()
    │ valida con PredictRequestSerializer
    ▼
PredictService.predict(model_name, hyperparams)
    │ delega
    ▼
ModelManager.predict(model_name, data)
    │ busca modelo en registry
    ▼
model_registry.get("pedido_sugerido")  →  PedidoSugeridoModel
    │ ejecuta predict
    ▼
PedidoSugeridoModel.predict(data)
    │ crea contexto + pipeline
    ▼
PredictContext + predict_pedido_sugerido_pipeline()
    │ ejecuta 10 steps en secuencia
    ▼
ctx.data_response = { "knn_xgb": [...], "apriori_xgb": [...], "destacados": [...] }
    │ retorna DTO
    ▼
PredictResponseDTO(predictions=..., success=True)
    │ serializa
    ▼
HTTP 200 { "success": true, "data": { ... }, "errors": [] }
```

### Entrenamiento

```
POST /api/v1/train/
{
  "model_name": "pedido_sugerido",
  "version": "1.0",
  "data_source": {
    "type": "sqlserver",
    "params": { "query": "SELECT * FROM VentasHistoricas" }
  }
}
```

```
TrainingView.post()
    │ valida con TrainRequestSerializer
    ▼
TrainingService.run(TrainRequestDTO)
    ▼
ModelManager.train(request)
    │ obtiene clase de pipeline del registry
    ├── get_pipeline("pedido_sugerido") → PedidoSugeridoPipeline
    │ construye datasource
    ├── DataSourceFactory.build(config) → SqlServerDataSourceStrategy
    │ instancia pipeline con datasource inyectado
    ├── PedidoSugeridoPipeline(data_source)
    │ crea contexto y ejecuta
    ▼
TrainingContext + pipeline.run(ctx)
    │ ejecuta 10 steps en secuencia
    ▼
TrainResponseDTO(steps_executed=[...], success=True)
    ▼
HTTP 200 { "success": true, "data": { "steps_executed": [...] } }
```

---

## 5. El sistema de Pipeline

Es el corazón de la arquitectura ML. Permite ejecutar secuencias de pasos complejos de forma ordenada, con manejo de errores y logging automático.

### PipelineBase

```
┌─────────────────────────────────────────────────┐
│                  PipelineBase[T]                │
│                                                 │
│  _steps: [StepABC, StepABC, StepABC, ...]       │
│                                                 │
│  run(ctx: T) → T:                               │
│    for step in _steps:                          │
│      if ctx.has_errors: DETENER                 │
│      ctx = step(ctx)                            │
│    return ctx                                   │
└─────────────────────────────────────────────────┘
```

### StepABC — Template Method Pattern

Cada paso implementa solo `execute()`. El método `__call__` es el **template** que agrega logging y registro de pasos automáticamente:

```
step.__call__(ctx)
    │
    ├── logger.info("step_started")
    ├── ctx = step.execute(ctx)       ← lógica concreta del paso
    ├── ctx.steps_executed.append(name)
    └── logger.info("step_finished")
```

```python
# Para crear un nuevo paso, solo hay que implementar execute():
class MiNuevoPaso(StepABC[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        # lógica...
        return ctx
```

### El contexto viaja por todos los pasos

```
ctx = TrainingContext(model_name="pedido_sugerido", version="1.0")
     │
     ▼
LoadDataStep      → ctx.raw_data = DataFrame(1M filas)
     │
     ▼
EdaCleanDataStep  → ctx.clean_data = DataFrame filtrado
     │
     ▼
ClusteringStep    → ctx.extra["model_km"] = modelo KMeans entrenado
     │
     ▼
KnnStep           → ctx.extra["model_knn"] = modelo KNN entrenado
     │
     ▼
...
     │
     ▼
SaveModelStep     → guarda .pkl en disco
     │
     ▼
RegistryModelStep → modelo disponible para predicciones
```

**Propagación de errores:** si un paso agrega a `ctx.errors`, el pipeline detiene la ejecución y lo reporta en el response. Ningún paso siguiente se ejecuta.

---

## 6. Patrones de diseño

### Strategy — Intercambiar comportamientos

Permite cambiar la fuente de datos o el tipo de modelo sin modificar el código que los usa.

```
DataSourceABC
    ├── CsvDataSourceStrategy       → pd.read_csv(path)
    └── SqlServerDataSourceStrategy → pd.read_sql(query, pyodbc)

MLModelABC
    └── PedidoSugeridoModel         → KNN + Apriori + RandomForest
    # └── OtroModelo                → cualquier otra lógica
```

### Factory — Construir objetos según configuración

`DataSourceFactory.build(config)` decide qué implementación crear según el `type` del config del request:

```python
# El pipeline recibe siempre DataSourceABC, sin importar la fuente
data_source = DataSourceFactory.build({"type": "csv", "params": {"path": "ventas.csv"}})
data_source = DataSourceFactory.build({"type": "sqlserver", "params": {"query": "..."}})
# Ambos son intercambiables para el pipeline
```

### Registry + Decorator — Auto-registro de pipelines

En lugar de mantener un diccionario hardcodeado de pipelines, cada pipeline se registra a sí mismo con un decorador:

```python
# infrastructure/ml/pipeline_registry.py
_TRAIN_PIPELINES: dict[str, type] = {}

def register_pipeline(model_name: str):
    def decorator(cls):
        _TRAIN_PIPELINES[model_name] = cls
        return cls
    return decorator

# training/pedido_sugerido/pipeline.py
@register_pipeline("pedido_sugerido")   # ← se registra solo
class PedidoSugeridoPipeline(PipelineBase):
    ...
```

El `ModelManager` solo llama `get_pipeline("pedido_sugerido")` y no sabe nada de qué pipelines existen. Para agregar un nuevo modelo, solo se decora su clase y se importa en `training/__init__.py`.

### Template Method — Estructura fija, comportamiento variable

`StepABC.__call__` define el flujo fijo (log → ejecutar → registrar). Las subclases solo implementan la lógica variable en `execute()`.

`PipelineBase.run()` define el flujo fijo (iterar pasos, detectar errores, loggear). Las subclases definen qué pasos se agregan en `__init__`.

### Singleton thread-safe — ModelRegistry

Un único registro global de modelos cargados, protegido con `threading.Lock()` para soportar múltiples requests simultáneos:

```python
model_registry = ModelRegistry()  # instancia global en domain/ml/model_registry.py
```

### Composite — Pipeline de pasos

`PipelineBase` compone una lista de `StepABC` y los trata a todos uniformemente con `step(ctx)`. Agregar o quitar pasos no requiere cambiar el orquestador.

---

## 7. Modelo pedido_sugerido

Recomienda productos a clientes basándose en tres fuentes:

| Fuente | Técnica | Descripción |
|---|---|---|
| `knn_xgb` | KNN + RandomForest | Productos que compran clientes similares, cantidad predicha por RF |
| `apriori_xgb` | Apriori + RandomForest | Productos relacionados por reglas de asociación, cantidad predicha por RF |
| `destacados` | Lista estática | Ofertas y liquidaciones configurables |

### Artefactos que guarda el modelo

Al entrenar, se serializa un `.pkl` con esta estructura:

```python
{
    "artefactos": {
        "model_knn": {
            "model": NearestNeighbors,     # vecinos cercanos
            "scaler": StandardScaler,       # normalización
            "enc_cat": OneHotEncoder,       # encoding categóricas
            "feats_num": [...],             # nombres features numéricas
            "cat_features": [...],          # nombres features categóricas
            "customers": [...],             # lista ordenada de cliente_ids
            "data": DataFrame,             # perfil agregado por cliente
            "perfil_pivot": DataFrame,     # matriz cliente × producto
        },
        "model_apriori": {
            "rules": DataFrame,            # reglas A→B con confidence, lift, support
        },
        "model_rf_cantidad": {
            "model": RandomForestRegressor, # predice cantidad vendida
            "encoder": OrdinalEncoder,      # encoding para RF
            "features": [...],              # nombres de features
        },
        "perfil_productos": DataFrame,     # historial completo de transacciones
    }
}
```

### Steps de entrenamiento

```
LoadDataStep              → carga datos crudos del datasource
EdaCleanDataStep          → limpia nulos, normaliza nombres de columnas
CalculoAtributosDerivadosStep → calcula 15+ features (recencia, frecuencia, etc.)
ClusteringKMeansStep      → segmenta clientes (k óptimo por Silhouette Score)
VecinosCercanosKnnStep    → entrena KNN (n óptimo por método del codo)
ConjuntoReglasAprioriStep → genera reglas de asociación entre productos
PrepareDataArbolesStep    → construye tabla de entrenamiento para RF
EnsembleArbolesRandomForestStep → entrena RF con RandomizedSearchCV
SaveModelStep             → serializa todos los artefactos a .pkl
RegistryModelStep         → registra modelo en memoria para predicciones
```

### Steps de predicción

```
LoadModelStep             → desempaqueta artefactos del .pkl cargado
ValidateClienteStep       → verifica cliente_id existe en historial
KnnFindNeighborsStep      → encuentra clientes similares con cosine distance
KnnBuildCandidatesStep    → construye candidatos desde historial de vecinos
KnnRankAndPredictStep     → asigna score + predice cantidad con RF
AprioriBuildCandidatesStep → construye candidatos desde reglas Apriori
AprioriRankAndPredictStep  → asigna score + predice cantidad con RF
ParetoFilterStep           → filtra por Pareto (top_n, cantidad_minima, %)
DestacadosStep             → agrega productos destacados
BuildResponseStep          → ensambla respuesta final
```

---

## 8. DataSources

Toda fuente de datos implementa `DataSourceABC`:

```python
class DataSourceABC(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame: ...
```

### Fuentes disponibles

**CSV** — para desarrollo local o datos históricos en planilla:
```json
{
  "type": "csv",
  "params": {
    "path": "ventas_2024.csv",
    "separator": ",",
    "encoding": "utf-8"
  }
}
```

**SQL Server** — para producción:
```json
{
  "type": "sqlserver",
  "params": {
    "query": "SELECT * FROM dbo.VentasHistoricas WHERE FechaVenta >= '2024-01-01'",
    "connection_string": "DRIVER=...;SERVER=...;DATABASE=..."
  }
}
```

Si no se pasa `connection_string`, se usa la configurada en `.env` (`ml_db_*`).

### Agregar una nueva fuente

1. Crear `app/infrastructure/ml/data_sources/mongo_data_source_strategy.py` que herede de `DataSourceABC`
2. Agregar el caso `"mongodb"` en `DataSourceFactory.build()`

---

## 9. Registro de modelos

`ModelRegistry` mantiene en memoria los modelos cargados. Es thread-safe con `threading.Lock()`.

### Ciclo de vida de un modelo

```
Startup (AppConfig.ready())
    └── load_initial_models()
          └── busca *.pkl en storage/models/
                └── PedidoSugeridoModel.load(path)
                      └── model_registry.register("pedido_sugerido", model)

Entrenamiento (POST /train/)
    └── RegistryModelStep
          └── carga nuevo .pkl
                └── model_registry.register("pedido_sugerido", model)
                      # reemplaza el anterior en memoria

Predicción (POST /predict/)
    └── model_registry.get("pedido_sugerido")
          └── PedidoSugeridoModel.predict(data)
```

---

## 10. Registro de pipelines

El decorador `@register_pipeline` permite que cada pipeline se registre a sí mismo. El `ModelManager` no necesita saber qué pipelines existen.

```
Importación de app.infrastructure.ml.training
    └── training/__init__.py importa PedidoSugeridoPipeline
          └── el decorador @register_pipeline("pedido_sugerido") se ejecuta
                └── _TRAIN_PIPELINES["pedido_sugerido"] = PedidoSugeridoPipeline

POST /train/ con model_name="pedido_sugerido"
    └── get_pipeline("pedido_sugerido")
          └── PedidoSugeridoPipeline
```

Para registrar un nuevo pipeline, solo se agrega una línea en `training/__init__.py`.

---

## 11. Configuración y settings

Hay dos sistemas de configuración separados:

| Archivo | Framework | Para qué |
|---|---|---|
| `config/settings.py` | Django | INSTALLED_APPS, middleware, REST_FRAMEWORK |
| `app/domain/core/config.py` | pydantic-settings | Variables de negocio (paths, DB, etc.) |

### Variables de entorno

Copiar `.env.example` a `.env` y completar:

```bash
SECRET_KEY=tu-clave-secreta-aqui

app_env=development
app_debug=true
log_level=INFO
timezone=America/La_Paz

path_data=storage/data
path_models=storage/models

ml_db_driver=ODBC Driver 17 for SQL Server
ml_db_server=tu-servidor
ml_db_database=tu-base-de-datos
ml_db_user=tu-usuario
ml_db_password=tu-contraseña

ALLOWED_HOSTS=localhost,127.0.0.1
```

`ml_db_connection_string` se construye automáticamente desde los campos individuales.

---

## 12. Logging

Se usa **structlog** con salida dual:

| Destino | Formato | Cuándo |
|---|---|---|
| Consola | Human-readable (colores) | `app_debug=true` |
| Consola | JSON | `app_debug=false` |
| `logs/app.log` | JSON siempre | Siempre, rotación diaria, 30 días |

### Uso

```python
from app.domain.core.logging import logger

logger.info("model_registered", name="pedido_sugerido", version="1.0")
logger.warning("cliente_sin_historial", cliente_id=14111)
logger.error("pipeline_failed", error=str(e))
```

Todos los logs incluyen automáticamente: timestamp con timezone, nivel, y nombre del logger.

---

## 13. Endpoints de la API

| Método | URL | Descripción |
|---|---|---|
| `GET` | `/api/v1/` | Health check |
| `POST` | `/api/v1/predict/` | Ejecutar predicción |
| `POST` | `/api/v1/train/` | Entrenar modelo |
| `GET` | `/api/v1/list_models/` | Listar modelos en memoria |
| `GET` | `/api/docs/` | Swagger UI (OpenAPI) |
| `GET` | `/api/redoc/` | ReDoc |

### Formato de respuesta

Todas las respuestas siguen el mismo envelope:

```json
{
  "success": true,
  "message": "Descripción de lo que pasó.",
  "data": { ... },
  "errors": [],
  "timestamp": "2026-04-05T12:00:00-04:00"
}
```

### Serializers por endpoint

Los serializers viven en `presentation/api/v1/endpoints/serializers/` organizados por caso de uso:

```
serializers/
  predict/
    request_serializer.py   → valida entrada del POST /predict/
    response_serializer.py  → documenta salida (OpenAPI)
  train/
    request_serializer.py   → valida entrada del POST /train/
    response_serializer.py  → documenta salida (OpenAPI)
  model/
    metadata_serializer.py  → documenta salida del GET /list_models/
```

---

## 14. Cómo agregar un nuevo modelo

Ejemplo: agregar un modelo `ventas_proyectadas`.

### Paso 1 — Crear el modelo ML

```python
# app/infrastructure/ml/models/ventas_proyectadas_model.py
from app.domain.ml.abstractions.ml_model_abc import MLModelABC

class VentasProyectadasModel(MLModelABC):
    def load(self, path: str) -> None:
        self._model = joblib.load(path)

    def predict(self, data: dict) -> dict:
        ctx = PredictContext(...)
        pipeline = ventas_proyectadas_predict_pipeline()
        ctx = pipeline.run(ctx)
        return ctx.data_response
```

### Paso 2 — Crear los pasos de entrenamiento

```python
# app/infrastructure/ml/training/ventas_proyectadas/steps.py
class LoadDataStep(StepABC[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        ...
        return ctx

class EntrenarModeloStep(StepABC[TrainingContext]):
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        ...
        return ctx
```

### Paso 3 — Crear el pipeline de entrenamiento con el decorador

```python
# app/infrastructure/ml/training/ventas_proyectadas/pipeline.py
from app.infrastructure.ml.pipeline_registry import register_pipeline

@register_pipeline("ventas_proyectadas")
class VentasProyectadasPipeline(PipelineBase):
    def __init__(self, data_source: DataSourceABC) -> None:
        super().__init__()
        self.add_step(LoadDataStep(data_source))
        self.add_step(EntrenarModeloStep())
        # ...
```

### Paso 4 — Registrar el import en `training/__init__.py`

```python
# app/infrastructure/ml/training/__init__.py
from app.infrastructure.ml.training.pedido_sugerido.pipeline import PedidoSugeridoPipeline  # noqa: F401
from app.infrastructure.ml.training.ventas_proyectadas.pipeline import VentasProyectadasPipeline  # noqa: F401
```

Eso es todo. `ModelManager` no necesita modificarse.

### Paso 5 — Crear el pipeline de predicción

```python
# app/infrastructure/ml/predict/ventas_proyectadas/pipeline.py
def ventas_proyectadas_predict_pipeline() -> PipelineBase:
    pipeline = PipelineBase()
    pipeline.add_step(MiPasoDePrediccion())
    return pipeline
```

---

## Dependencias principales

| Librería | Versión | Uso |
|---|---|---|
| Django | 5.2 | Framework web |
| djangorestframework | 3.16.0 | API REST |
| drf-spectacular | 0.28.0 | Swagger / OpenAPI |
| pydantic-settings | 2.13.1 | Configuración tipada |
| structlog | 25.5.0 | Logging estructurado |
| scikit-learn | 1.8.0 | KMeans, KNN, RandomForest |
| mlxtend | 0.24.0 | Apriori, association rules |
| pandas | 3.0.1 | Manipulación de datos |
| joblib | 1.5.3 | Serialización de modelos |
| pyodbc | — | Conexión SQL Server |
