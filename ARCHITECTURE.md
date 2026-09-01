# Arquitectura actual — xApiPredict

Este documento describe el estado **implementado** del repositorio a septiembre de 2026. No es una arquitectura objetivo: señala también los límites y desalineaciones que existen hoy entre las capas y los contratos HTTP.

## 1. Propósito y componentes

`xApiPredict` es una aplicación Django 5.2 que entrena y sirve predicciones del modelo `pedido_sugerido`. Además de la API REST, incluye una UI HTML mínima, persistencia de versiones y resultados, y un planificador de trabajos basado en APScheduler.

```text
Clientes HTTP / navegador
        │
        ├── Django REST Framework: /api/v1/
        ├── HTML: / y /models/
        └── Admin Django: /admin/
                │
                ▼
    Presentación → servicios / casos de uso → modelos ML y trabajos
                │                                  │
                │                                  ├── registro de modelos en memoria
                │                                  ├── pipelines de entrenamiento/predicción
                │                                  └── adaptadores de lectura/escritura
                ▼
      ORM Django / repositorios ────────────────► SQLite o SQL Server
```

El diseño está inspirado en Clean Architecture, pero no es una separación estricta: los modelos de persistencia Django residen en `app/domain/models` y el `domain` importa Django y pandas. Por ello, las carpetas expresan responsabilidades, no fronteras libres de framework.

## 2. Estructura del repositorio

```text
config/
  settings.py                    Configuración Django y DRF
  urls.py                         Rutas raíz, OpenAPI, web y admin
  app_startup_middleware.py       Carga de modelos y scheduler al primer request

app/
  domain/
    abstractions/                 Contratos: modelo, pipeline, step, fuente, writer y repo
    core/                         Settings Pydantic, logging, excepciones y hora local
    dtos/                         DTOs de entrenamiento, predicción y envelope
    ml/                           Contextos, registry y metadata de modelo
    models/                       Modelos Django para el esquema ml
  application/
    services/                     Orquestación de modelos, fuentes, jobs y resultados
    ml/                           Manager, predictor y pipelines concretos
    jobs/                         Registry, runner y handlers de tareas
    utils/                        Conversión de parámetros y versión por fecha
  infrastructure/
    db/                           Configuración Django DB, migraciones y repositorios ORM
    data_sources/                 Adaptadores CSV y SQL Server
    data_writers/                 Escritura batch en SQL Server
    jobs/                         Scheduler APScheduler
  presentation/
    api/                          Envelope y handler global de excepciones
    api/v1/                       Views, rutas y serializers DRF
    web/                          Vistas y templates HTML
  apps.py                         Inicializa structlog desde AppConfig.ready()

storage/
  models/                         Artefactos joblib (.pkl) de los modelos entrenados
  data/                           Archivos CSV que se usan como fuentes locales
logs/                             Salida de structlog (si la configuración la habilita)
```

## 3. Responsabilidades y dependencias

| Área | Responsabilidad actual | Dependencias relevantes |
|---|---|---|
| `presentation` | Convierte HTTP a llamadas de servicio, valida serializers y devuelve respuestas DRF. | DRF, servicios de aplicación |
| `application` | Coordina entrenamiento, inferencia, lotes y ejecución de tareas. Alberga los pipelines ML concretos. | Dominio, repositorios y adaptadores concretos |
| `domain` | Define contratos, DTOs, contextos, registry y entidades persistentes. | Django ORM, pandas, Pydantic y structlog en algunos módulos |
| `infrastructure` | Implementa acceso a DB, fuentes de datos, writers y scheduler. | Django ORM, pyodbc, APScheduler |

La dirección de dependencias no es totalmente unidireccional. Por ejemplo, `ModelManager` (aplicación) instancia `DataSourceFactory` y servicios que usan repositorios concretos; `StepABC` (dominio) importa un servicio de aplicación de forma diferida para persistir el log de cada paso. Estas decisiones son parte del comportamiento actual y se deben considerar si se planifica una separación más estricta.

## 4. Arranque y ciclo de vida

1. Django carga `config.settings`, que inicializa `DATABASES` mediante `get_databases()`.
2. `AppConfig.ready()` configura `structlog` y registra el evento `startup`.
3. En la construcción del primer middleware, `AppStartupMiddleware` ejecuta una sola vez por proceso:
   - `load_initial_models()` consulta las `VersionModelo` activas, carga su `.pkl` y las registra en memoria;
   - inicia `JobScheduler` cuando `RUN_MAIN == "true"` o la aplicación no está en modo debug.

El registro se rellena desde versiones activas de la base de datos; no recorre automáticamente `storage/models`. El scheduler es un `BackgroundScheduler` embebido en el proceso web, por lo que cada proceso que satisfaga la condición puede tener su propia instancia.

## 5. Configuración y persistencia

### Configuración

Hay dos fuentes de configuración:

| Componente | Uso |
|---|---|
| `config/settings.py` | Django, rutas, middleware, DRF, OpenAPI y templates. |
| `app/domain/core/config.py` | Variables `.env` tipadas: entorno, logging, paths, timezone y conexiones. |

La base de datos del ORM usa SQL Server (`mssql-django`) cuando `app_db_server` tiene valor; si no, usa `db.sqlite3` local. Las migraciones están bajo `app.infrastructure.db.migrations` y crean tablas en el esquema SQL Server `ml`.

Las fuentes SQL Server de los pipelines no reutilizan automáticamente `ml_db_*`: requieren `driver`, `server`, `database` y credenciales —o `trusted_connection`— en los parámetros almacenados de la fuente de datos.

### Entidades persistidas

Todas heredan de `BaseModelABC`, que agrega `id`, auditoría básica y borrado lógico (`eliminado`). Los repositorios filtran normalmente por `eliminado=False`.

```text
FuenteDatos 1 ─── * FuenteDatosParametros
     │
     ├── * TareaProgramada 1 ─── * TareaParametro
     │             │
     │             └── * EjecucionTareaProgramada 1 ─── * LogTareaProgramada
     │
     └── * VersionModelo (solo una relación de fuente por versión)

VersionModelo 1 ─── * LotePrediccion 1 ─── * ResultadoPrediccion
```

`VersionModelo` guarda el nombre, versión, ruta del `.pkl`, parámetros y estado activo. Al concluir un entrenamiento correcto, el servicio desactiva las versiones anteriores del mismo modelo y crea una nueva versión activa. `LotePrediccion` y `ResultadoPrediccion` persisten los resultados de ejecuciones masivas.

## 6. API, web y manejo de errores

### Rutas registradas

| Método | Ruta | Implementación | Respuesta actual |
|---|---|---|---|
| `GET` | `/` | `HomeView` | Template HTML `app/home.html`. |
| `GET` | `/models/` | `web.ModelsView` | Template HTML con modelos registrados. |
| `GET` | `/api/v1/` | `HealthView` | JSON directo: `{ "status": "ok" }`. |
| `POST` | `/api/v1/predict/` | `PredictView` | Envelope estándar con `PredictResponseDTO`. |
| `POST` | `/api/v1/train/` | `TrainingView` | Ruta expuesta; ver desalineaciones. |
| `GET` | `/api/v1/list_models/` | `ModelsView` | Envelope con metadata del registry. |
| `GET` | `/api/schema/` | drf-spectacular | Esquema OpenAPI. |
| `GET` | `/api/docs/` / `/api/redoc/` | drf-spectacular | Swagger UI / ReDoc. |

Excepto el health check, las views REST usan `success_response()` y devuelven:

```json
{
  "success": true,
  "message": "...",
  "data": {},
  "errors": [],
  "timestamp": "2026-09-01T..."
}
```

`api_exception_handler` transforma errores DRF conocidos y excepciones no controladas a ese mismo envelope. Los serializers solo se usan para validación y documentación: el payload exitoso se convierte desde dataclasses por `responses._serialize`.

No hay endpoints registrados para administrar fuentes de datos, tareas, clientes, versiones, lotes o resultados. Esas capacidades existen en servicios y repositorios, y actualmente se invocan desde jobs o código interno. `admin.py` importa las entidades, pero no las registra con `admin.site.register`, por lo que el admin no las expone por ese archivo.

## 7. Modelo en memoria y predicción individual

`ModelRegistry` es una instancia global protegida con `threading.Lock`. Acepta reemplazar un modelo registrado por defecto y expone metadatas para listar modelos. Solo registra instancias de `MLModelABC` que estén cargadas.

```text
POST /api/v1/predict/
  → PredictView
  → PredictService
  → ModelManager.predict()
  → ModelRegistry.get(nombre)
  → PedidoSugerido.predict(parámetros)
  → PedidoSugeridoPredictPipeline.run(PredictContext)
  → lista de recomendaciones
```

El único predictor concreto es `app.application.ml.predictors.pedido_sugerido.PedidoSugerido`. Carga un diccionario joblib con `artefactos` y crea un `PredictContext`; no existe una capa `infrastructure/ml/models` como indicaba la documentación anterior.

El pipeline de predicción construye siempre los diez pasos siguientes:

1. `LoadModelStep`
2. `ValidateClienteStep`
3. `KnnFindNeighborsStep`
4. `KnnBuildCandidatesStep`
5. `KnnRankAndPredictStep`
6. `ParetoFilterStep`
7. `AprioriBuildCandidatesStep`
8. `AprioriRankAndPredictStep`
9. `DestacadosStep`
10. `BuildResponseStep`

Combina vecinos cercanos con un regresor `RandomForestRegressor` para cantidad, reglas Apriori y destacados estáticos. Apriori usa como antecedentes las recomendaciones KNN filtradas por Pareto. Las opciones `recomendacion_apriori` y `recomendacion_destacados` deciden si esas listas se incluyen en la respuesta final. La salida es una lista plana de resultados normalizados por `armar_respuesta`, no un objeto con secciones `knn_xgb`, `apriori_xgb` y `destacados`.

## 8. Entrenamiento y artefactos

El entrenamiento se ejecuta correctamente desde un `TrainRequestDTO` asociado a una `TareaProgramada` (usado por los handlers). `ModelManager` toma los parámetros de la tarea, obtiene la fuente por `fuente_datos_id`, construye el datasource registrado y ejecuta el pipeline de entrenamiento registrado.

```text
TareaProgramada + FuenteDatos
  → ModelManager.train()
  → DataSourceFactory.build(tipo, parámetros)
  → PedidoSugeridoPipeline.set_datasource()
  → TrainingContext
  → pipeline.run()
  → .pkl + ModelRegistry + VersionModelo activa
```

Los pasos del pipeline `pedido_sugerido` son:

1. `LoadDataStep`
2. `EdaCleanDataStep`
3. `CalculoAtributosDerivadosStep`
4. `ClusteringKMeansStep`
5. `VecinosCercanosKnnStep`
6. `ConjuntoReglasAprioriStep`
7. `PrepareDataArbolesStep`
8. `EnsembleArbolesRandomForestStep`
9. `SaveModelStep`
10. `RegistryModelStep`

El `.pkl` generado contiene `model_km`, `model_knn`, `model_apriori`, `model_rf_cantidad` e `historial_ventas`, agrupados en `artefactos`. Se guarda con un nombre `modelo_<nombre>_<versión>.pkl`; la versión la genera `ObtenerVersion()` a partir de la fecha actual (`YYYY.MM.DD`).

`PipelineBase.run()` reconstruye la lista de pasos en cada ejecución y la corta si el contexto contiene errores. `StepABC.__call__()` mide duración, captura excepciones, agrega el paso ejecutado y, cuando existe `ejecucion_id`, persiste un `LogTareaProgramada`.

## 9. Fuentes y destinos de datos

Las implementaciones se registran por decorador en registries en memoria y se resuelven mediante factories.

| Contrato | Tipo registrado | Implementación | Uso |
|---|---|---|---|
| `DataSourceABC` | `csv` | `CsvDataSourceStrategy` | Lee `settings.path_data + path` con pandas. |
| `DataSourceABC` | `sqlserver` | `SqlServerDataSourceStrategy` | Ejecuta una query con pyodbc y lee por lotes de 5.000 filas. |
| `DataWriterABC` | `sqlserver` | `SqlServerBatchWriter` | Inserta un DataFrame con `fast_executemany`, transacción y lotes configurables. |

Los parámetros de conexión y de la query se almacenan como pares clave/valor en `FuenteDatosParametros`. No se tipan al recuperarse: el repositorio devuelve `dict[str, str]`; cada consumidor convierte lo que necesita.

## 10. Tareas programadas

`JobScheduler` consulta las `TareaProgramada` activas con expresión cron y registra un job por tarea. Ante un fallo, programa reintentos con `DateTrigger` hasta `max_reintentos`, con espera `delay_reintento_segundos`.

```text
APScheduler → JobService.ejecutar()
  → crea EjecucionTareaProgramada (ejecutando)
  → JobRunner → handler registrado por TipoJob
  → marca exitosa o fallida
  → opcionalmente programa reintento
```

Handlers registrados:

| Tipo | Estado |
|---|---|
| `training` | Entrena y falla la ejecución si el DTO de respuesta indica error. |
| `training_predict` | Entrena, crea un lote, recorre todos los clientes y persiste resultados. |
| `consulta_sql` | Lee una fuente y escribe el DataFrame a SQL Server. |
| `predict` | Registrado pero sin implementación; su función no acepta `ejecucion_id`, mientras que `JobRunner` lo envía. Actualmente fallaría al ejecutarse. |

Las rutas web y REST no exponen acciones para crear, actualizar o ejecutar estas tareas; `JobScheduler` sí contiene métodos internos para ello.

## 11. Patrones empleados

| Patrón | Aplicación |
|---|---|
| Strategy + Factory | Fuentes y writers se registran y se construyen según su tipo. |
| Registry + Decorator | Pipelines y handlers se auto-registran al importar sus módulos. |
| Template Method | `StepABC.__call__()` controla log, medición, captura y persistencia; cada paso implementa `execute()`. |
| Composite | `PipelineBase` trata una secuencia de pasos con la misma interfaz. |
| Repository | Repositorios ORM encapsulan las consultas y el borrado lógico. |
| Singleton práctico | `model_registry`, servicios, manager, scheduler y runner son instancias globales de módulo. |

## 12. Desalineaciones y deuda técnica observada

Estas observaciones son importantes para quien extienda o consuma la API:

1. **Contrato de entrenamiento roto.** `TrainingView` valida `model_name`, `version` y `data_source`, pero construye `TrainRequestDTO` con esos mismos argumentos. El DTO actual solo declara `tarea_programada`, `parameters` y `ejecucion_id`; por tanto el endpoint lanzará `TypeError` antes de entrenar. El camino respaldado por el código es el handler `training`, que recibe una `TareaProgramada` con parámetros y fuente ya persistidas.
2. **OpenAPI de modelos desactualizado.** `ModelMetadataSerializer` documenta `feature_names`, `hyperparams` y `trained_at`, mientras que `ModelMetadata` contiene `parameters`, `loaded_at`, `extra` y `path_model`. La documentación de `/list_models/` no representa fielmente el objeto serializado.
3. **Metadata de inferencia limitada.** Al cargar o registrar un modelo, solo se asignan nombre, versión y ruta. Los parámetros/extra no se restauran desde `VersionModelo`, aunque `PedidoSugerido.predict()` los copia al contexto.
4. **Modelos Django dentro de domain.** Esto impide considerar esa capa independiente de infraestructura o de Django sin una refactorización posterior.
5. **Seguridad de secretos en logs.** Los builders SQL Server registran el connection string construido, que puede contener contraseña. Es un riesgo operativo a corregir antes de producción.
6. **Admin sin registros.** El módulo importa las entidades, pero no las registra; el panel no ofrece gestión de esos modelos por la configuración actual.

## 13. Cómo extender el sistema

Para una nueva fuente de datos, implementar `DataSourceABC`, registrar un builder con `@register_datasource("tipo")` y asegurar que su módulo se importe desde `app.infrastructure.data_sources.__init__`.

Para un writer, aplicar el mismo esquema con `DataWriterABC` y `@register_datawriter`.

Para un nuevo modelo entrenable:

1. Crear un predictor que implemente `MLModelABC`.
2. Crear un `TrainingPipelineBase` y registrar su clase con `@register_pipeline("nombre")`.
3. Importar el módulo del pipeline desde `app.application.ml.pipelines.training.__init__` para activar el registro.
4. Implementar el pipeline de predicción y ajustar `load_initial_models()` —hoy instancia `PedidoSugerido` para todas las versiones activas— para resolver el predictor correcto.
5. Crear la tarea, fuente de datos y parámetros persistidos que usará el job de entrenamiento.

Antes de habilitarlo por HTTP, alinear el serializer, el DTO y la view de entrenamiento, y actualizar los serializers de respuesta para que OpenAPI refleje los objetos reales.
