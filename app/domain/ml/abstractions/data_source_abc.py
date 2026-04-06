from abc import ABC, abstractmethod

import pandas as pd


class DataSourceABC(ABC):
    @abstractmethod
    def load(self) -> pd.DataFrame: ...
