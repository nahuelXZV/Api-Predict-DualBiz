from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame:
        ...
