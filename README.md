# Api-Predict-DualBiz

API de predicción para DualBiz desarrollada en Python.

## Requisitos

- Python 3.13.12
- pip

## Instalación y Uso

1. **Clonar el repositorio:**
```bash
git clone https://github.com/nahuelXZV/Api-Predict-DualBiz.git
cd Api-Predict-DualBiz
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
venv\Scripts\activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación:**
```bash
uvicorn app.main:app --reload
```

La API estará disponible en `http://localhost:8000/docs`
