import numpy as np
import pandas as pd
import xgboost as xgb
from app.domain.core.logging import logger
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import RandomizedSearchCV


def calcular_nro_clusters_kmeans(
    data: np.ndarray, min_clusters: int = 2, max_clusters: int = 8
) -> int:
    """
    Determina el número óptimo de clusters para KMeans usando el Silhouette Score.

    Entrena un KMeans para cada valor de k en el rango [min_clusters, max_clusters]
    y evalúa qué tan bien separados están los clusters mediante el Silhouette Score,
    que mide la cohesión interna vs la separación entre clusters (rango: -1 a 1,
    más alto es mejor). Elige el k con el score más alto.

    Args:
        data: Matriz de features ya escalada (n_samples, n_features).
        min_clusters: Número mínimo de clusters a evaluar (default 2).
        max_clusters: Número máximo de clusters a evaluar (default 8).

    Returns:
        int: Número óptimo de clusters según el Silhouette Score más alto.
    """
    mejor_k = min_clusters
    mejor_score = -1

    for k in range(min_clusters, max_clusters + 1):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init="auto")
        labels = km.fit_predict(data)
        score = silhouette_score(data, labels)

        if score > mejor_score:
            mejor_score = score
            mejor_k = k

    logger.info("kmeans_clusters_optimos", k=mejor_k, silhouette=round(mejor_score, 4))
    return mejor_k


def calcular_nro_vecinos_knn(data: np.ndarray, k_min: int = 3, k_max: int = 51) -> int:
    """
    Determina el número óptimo de vecinos para KNN usando el método del codo
    sobre la curva de distancias promedio.

    Entrena un NearestNeighbors para cada k impar en el rango [k_min, k_max]
    y calcula la distancia coseno promedio de cada punto a sus k vecinos reales
    (excluyendo la distancia a sí mismo). A medida que k crece, la distancia
    promedio aumenta — el codo indica el punto donde agregar más vecinos ya no
    aporta similitud relevante. Se detecta automáticamente como el máximo de la
    segunda derivada de la curva.

    Args:
        data: Matriz de features ya escalada (n_samples, n_features).
        k_min: Número mínimo de vecinos a evaluar (default 3).
        k_max: Número máximo de vecinos a evaluar (default 51, se ajusta al tamaño del dataset).

    Returns:
        int: Número óptimo de vecinos según el codo de la curva de distancias.
    """
    k_max = min(k_max, len(data) - 1)
    k_range = list(range(k_min, k_max + 1, 2))
    distancias_promedio = []

    for k in k_range:
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(data)
        distancias, _ = nn.kneighbors(data)

        # Mejor: promedio de TODAS las distancias
        distancias_promedio.append(distancias[:, 1:].mean())

    segunda_derivada = np.diff(np.diff(distancias_promedio))
    idx = np.argmax(segunda_derivada)

    k_optimo = k_range[idx + 2]

    logger.info("knn_vecinos_optimos", k=k_optimo)
    return k_optimo


def calcular_mejores_params_xgb(
    X: pd.DataFrame,
    y: pd.Series,
    n_iter: int = 30,
    cv: int = 3,
    random_state: int = 42,
) -> dict:
    """
    Busca los mejores hiperparámetros para XGBRegressor usando búsqueda
    aleatoria con validación cruzada (RandomizedSearchCV).

    En lugar de probar todas las combinaciones posibles (GridSearch),
    muestrea aleatoriamente `n_iter` combinaciones del espacio de búsqueda
    y evalúa cada una con `cv` folds. Devuelve la combinación con menor
    error cuadrático medio (RMSE). Es el enfoque estándar cuando el espacio
    de hiperparámetros es grande y GridSearch sería prohibitivo.

    Args:
        X: DataFrame de features ya codificadas (sin valores nulos).
        y: Serie con el target (cantidad_vendida).
        n_iter: Número de combinaciones aleatorias a evaluar (default 30).
                Más alto = más preciso pero más lento.
        cv: Número de folds para la validación cruzada (default 3).
        random_state: Semilla para reproducibilidad (default 42).

    Returns:
        dict: Mejores hiperparámetros encontrados, listos para pasar
              directamente a XGBRegressor(**resultado).
    """
    param_space = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5, 6, 8],
        "subsample": [0.6, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    model = xgb.XGBRegressor(random_state=random_state, verbosity=0)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
    )
    search.fit(X, y)

    logger.info("xgb_params_optimos", params=search.best_params_, rmse_cv=round(-search.best_score_, 4))
    return search.best_params_


def calcular_params_apriori(canastas: list[list]) -> dict:
    """
    Determina automáticamente min_support, min_confidence y min_lift para Apriori
    a partir de las estadísticas de las canastas de compra.

    Estrategia:
        - min_support: percentil 20 de los soportes individuales por producto,
          acotado entre 0.02 y 0.20. Captura el 80% de los productos más frecuentes
          sin incluir los muy raros que generan ruido.
        - min_confidence: 2× el soporte medio de los productos, acotado entre
          0.15 y 0.60. Exige que la regla sea al menos el doble de probable que
          encontrar el consecuente por azar.
        - min_lift: fijo en 1.0. Es el mínimo matemáticamente significativo
          (lift < 1 indica asociación negativa) y no tiene sentido optimizarlo.

    Args:
        canastas: Lista de listas, una por cliente, con los productos que compró.

    Returns:
        dict con claves min_support, min_confidence, min_lift.
    """
    n_clientes = len(canastas)
    conteo = {}
    for canasta in canastas:
        for producto in canasta:
            conteo[producto] = conteo.get(producto, 0) + 1

    soportes = np.array([v / n_clientes for v in conteo.values()])

    min_support = float(np.clip(np.percentile(soportes, 20), 0.02, 0.20))
    min_confidence = float(np.clip(2 * soportes.mean(), 0.15, 0.60))
    min_lift = 1.0

    params = {
        "min_support": round(min_support, 3),
        "min_confidence": round(min_confidence, 3),
        "min_lift": min_lift,
    }
    logger.info("apriori_params_calculados", **params)
    return params
