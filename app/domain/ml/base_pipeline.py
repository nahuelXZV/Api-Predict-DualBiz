from typing import Generic, TypeVar

from app.domain.ml.base_step import BaseStep
from app.domain.core.logging import logger

from app.domain.ml.base_context import BaseContext
 
T = TypeVar("T", bound=BaseContext) 

class BasePipeline(Generic[T]):

    def __init__(self) -> None:
        self._steps = []

    def add_step(self, step: BaseStep[T]) -> None:
        self._steps.append(step)

    def run(self, ctx: T) -> T:
        step_names = [s.name for s in self._steps]

        logger.info(
            "pipeline_started",
            model = ctx.model_name,
            steps = step_names,
        )

        for step in self._steps:
            print(f"Ejecutando step: {step.name}")
            if ctx.has_errors:
                skipped = [
                    s.name for s in self._steps
                    if s.name not in ctx.steps_executed
                ]
                logger.warning(
                    "pipeline_aborted",
                    model   = ctx.model_name,
                    reason  = ctx.errors,
                    skipped = skipped,
                )
                break

            ctx = step(ctx)

        logger.info(
            "pipeline_finished",
            model   = ctx.model_name,
            success = not ctx.has_errors,
            steps   = ctx.steps_executed,
            errors  = ctx.errors,
        )

        return ctx


    @property
    def steps(self) -> list[BaseStep[T]]:
        return list(self._steps)