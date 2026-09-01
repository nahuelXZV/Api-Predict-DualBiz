# xApiPredict

Aplicación Django para generar sugerencias de pedido por cliente. El modelo actual, `pedido_sugerido`, combina KNN, reglas de asociación Apriori y `RandomForestRegressor` para estimar cantidades. También conserva versiones de modelos, ejecuta tareas programadas y persiste resultados masivos.

Para la estructura técnica, los flujos internos y las decisiones de arquitectura, consultá [ARCHITECTURE.md](ARCHITECTURE.md).

## Requisitos

- Python 3.13+
- pip
- SQL Server con un driver ODBC compatible
- Git, si se clona el repositorio

Las migraciones crean el esquema `ml` con SQL específico de SQL Server. Aunque la configuración declara un fallback a SQLite, una instalación limpia con las migraciones actuales requiere SQL Server o adaptar previamente esa migración.

## Puesta en marcha

```bash
git clone https://github.com/nahuelXZV/Api-Predict-DualBiz.git
cd Api-Predict-DualBiz
python -m venv .venv
```

Activá el entorno e instalá las dependencias:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

Copiá `.env.example` a `.env` y configurá la conexión del ORM:

```env
SECRET_KEY=reemplazar-por-una-clave-segura
app_env=development
app_debug=true
timezone=America/La_Paz

path_data=storage/data
path_models=storage/models

app_db_driver=ODBC Driver 17 for SQL Server
app_db_server=servidor-sql
app_db_database=base_de_datos
app_db_user=usuario
app_db_password=contraseña

ALLOWED_HOSTS=localhost,127.0.0.1
```

Luego ejecutá:

```bash
python manage.py migrate
python manage.py runserver
```

## Operación esencial

Para servir predicciones debe existir una `VersionModelo` activa en la base de datos y el archivo `.pkl` asociado debe estar disponible. Al iniciar, la aplicación carga esas versiones en memoria.

El entrenamiento se opera mediante una `TareaProgramada` asociada a una `FuenteDatos`:

- `training` entrena y activa una nueva versión del modelo.
- `training_predict` entrena, procesa clientes y guarda un lote de resultados.
- `consulta_sql` mueve datos desde una fuente a SQL Server por lotes.

Las fuentes soportadas son `csv` y `sqlserver`. Las tareas activas con una expresión cron se registran en APScheduler al iniciar el proceso.

Actualmente la administración de fuentes, tareas, clientes, versiones y lotes se realiza mediante el ORM, scripts internos o SQL controlado; no hay una interfaz de gestión lista para esas entidades.

## Estado actual

- El flujo de entrenamiento operativo es el de tareas programadas.
- La ruta HTTP de entrenamiento requiere alinear su serializer, view y DTO antes de usarse.
- El panel admin permite autenticación, pero todavía no registra los modelos de dominio para administrarlos.
- Evitá incluir contraseñas en parámetros que puedan quedar registrados en logs.
