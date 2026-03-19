from abc import ABC, abstractmethod

from app.domain.core.logging import logger
from app.domain.ml.base_context import TrainingContext

class BaseTrainingStep(ABC):
    @abstractmethod
    def execute(self, ctx: TrainingContext) -> TrainingContext:
        ...

    def __call__(self, ctx: TrainingContext) -> TrainingContext:
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