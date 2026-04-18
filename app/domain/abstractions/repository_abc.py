from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class RepositoryABC(ABC, Generic[T]):
    @abstractmethod
    def get_by_id(self, id: int) -> T | None: ...

    @abstractmethod
    def delete(self, id: int) -> None: ...

    @abstractmethod
    def list_all(self) -> list[T]: ...

    @abstractmethod
    def save(self, entity: T) -> None: ...

    @abstractmethod
    def update(self, id: int, entity: T) -> None: ...

    @abstractmethod
    def exists(self, id: int) -> bool: ...

    @abstractmethod
    def create(self, **kwargs) -> T: ...

    @abstractmethod
    def deactivate_all(self, model_name: str) -> None: ...