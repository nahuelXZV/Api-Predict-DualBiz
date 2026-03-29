from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from app.domain.core.logging import logger
from app.domain.ml.base_context import BaseContext

T = TypeVar("T", bound=BaseContext) 

class BaseStep(ABC, Generic[T]):
    @abstractmethod
    def execute(self, ctx: T) -> T:
        ...

    def __call__(self, ctx: T) -> T:
        logger.info(
            "step_started",
            step  = self.name,
            model = ctx.model_name,
        )

        ctx = self.execute(ctx)
        ctx.steps_executed.append(self.name)

        logger.info(
            "step_finished",
            step  = self.name,
            model = ctx.model_name,
        )

        return ctx

    @property
    def name(self) -> str:
        return self.__class__.__name__