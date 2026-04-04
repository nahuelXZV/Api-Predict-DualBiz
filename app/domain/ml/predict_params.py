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
    candidatos: list
    cliente_id: Any
    perfil_productos: pd.DataFrame
    segmento: int
    fuente_nueva: str = "vecinos"
