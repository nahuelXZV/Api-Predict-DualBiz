# Api-Predict-DualBiz

API REST de predicción y entrenamiento de modelos de Machine Learning, desarrollada con Django. Genera recomendaciones de pedidos sugeridos para clientes combinando KNN, reglas de asociación Apriori y Random Forest.

---
## Requisitos previos

- Python 3.13+
- pip
- Git

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/nahuelXZV/Api-Predict-DualBiz.git
cd Api-Predict-DualBiz
```

### 2. Crear y activar entorno virtual

```bash
# Crear
python -m venv venv

# Activar — Windows
venv\Scripts\activate || .venv\Scripts\activate 

# Activar — Linux / macOS
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
```

Editá `.env` con tus valores:

```env
SECRET_KEY=django-insecure-reemplaza-esto-con-una-clave-segura

app_name=ApiPredict
app_version=1.0.0
app_env=development
app_debug=true

log_level=INFO
timezone=America/La_Paz

path_data=storage/data
path_models=storage/models

ALLOWED_HOSTS=localhost,127.0.0.1
```

### 5. Aplicar migraciones

```bash
python manage.py migrate
```

### 6. Ejecutar el servidor

```bash
python manage.py runserver
```

La API estará disponible en `http://localhost:8000`

---
## Estructura del proyecto

```
xApiPredict/
├── config/                        # Configuración Django
│   ├── settings.py
│   └── urls.py
├── app/
│   ├── presentation/              # Capa de presentación (HTTP)
│   │   ├── api/                   # Endpoints REST
│   │   │   ├── responses.py       # Envelope de respuestas
│   │   │   ├── exception_handler.py
│   │   │   └── v1/endpoints/      # Views + Serializers
│   │   └── web/                   # Vistas HTML
│   │       ├── views/
│   │       └── templates/
│   ├── application/               # Casos de uso (Services)
│   │   └── services/
│   ├── domain/                    # Abstracciones y contratos
│   │   ├── core/                  # Config, logging, exceptions
│   │   ├── dtos/                  # Data Transfer Objects
│   │   └── ml/                    # Base classes: BaseModel, BaseStep, BasePipeline
│   └── infrastructure/            # Implementaciones concretas
│       └── ml/
│           ├── model_manager.py   # Orquestador principal
│           ├── load_models.py     # Carga al inicio
│           ├── models/            # PedidoSugeridoModel
│           ├── training/          # Pipeline de entrenamiento (11 steps)
│           └── predict/           # Pipeline de predicción (10 steps)
├── storage/
│   ├── data/                      # Dataset CSV de entrenamiento
│   └── models/                    # Modelos .pkl entrenados
├── logs/                          # Logs rotativos diarios
├── manage.py
├── requirements.txt
└── .env.example
```

---
```bash
# Ver logs en tiempo real
tail -f logs/app.log
```

---

## Crear superusuario (admin)

```bash
python manage.py createsuperuser
```

Panel disponible en `http://localhost:8000/admin/`
