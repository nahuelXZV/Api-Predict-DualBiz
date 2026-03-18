from app.domain.ml.base_step import BaseTrainingStep
from app.core.logging import logger

from app.domain.ml.context import TrainingContext

class TrainingPipeline:
    def __init__(self, steps: list[BaseTrainingStep]) -> None:
        if not steps:
            raise ValueError("TrainingPipeline: la lista de steps no puede estar vacía.")
        self._steps = steps

    def run(self, ctx: TrainingContext) -> TrainingContext:
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
    def steps(self) -> list[BaseTrainingStep]:
        return list(self._steps)