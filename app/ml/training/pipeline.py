from app.domain.base_step import BaseTrainingStep
from app.ml.training.context import TrainingContext
from app.core.logging import logger

class TrainingPipeline:
    """
    Ejecuta una lista ordenada de BaseTrainingStep.

    Comportamiento:
      - Si un step agrega errors al contexto, el pipeline se detiene
        antes de ejecutar el siguiente step.
      - Esto permite que EvaluateStep aborte antes de llegar a RegisterStep
        cuando el modelo no alcanza el umbral de calidad.

    Uso:
        pipeline = TrainingPipeline(steps=[
            IngestStep(),
            CleanStep(),
            FeatureEngStep(target_column="species"),
            TrainStep(),
            EvaluateStep(min_accuracy=0.90),
            RegisterStep(model_class=IrisClassifier),
        ])
        result = pipeline.run(ctx)
    """

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
            # Si un step anterior agregó errores, abortar
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
            metrics = ctx.metrics,
            errors  = ctx.errors,
        )

        return ctx


    @property
    def steps(self) -> list[BaseTrainingStep]:
        return list(self._steps)