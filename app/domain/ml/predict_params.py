from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ParetoConfig:
    top_n: int
    cantidad_minima: float
    porcentaje_volumen: float


@dataclass
class BuildFeaturesRequest:
    candidatos: list[str]
    cliente_id: Any
    historial_ventas: pd.DataFrame
    segmento: int
    fuente_nueva: str = "vecinos"
