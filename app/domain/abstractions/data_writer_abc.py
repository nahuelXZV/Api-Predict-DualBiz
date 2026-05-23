from abc import ABC, abstractmethod

import pandas as pd


class DataWriterABC(ABC):
    @abstractmethod
    def insert_many(self, data: pd.DataFrame) -> int:
        ...
