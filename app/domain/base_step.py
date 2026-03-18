from abc import ABC, abstractmethod

from app.ml.training.context import TrainingContext
from app.core.logging import logger

class BaseTrainingStep(ABC):
    """
    ABC para todos los steps del pipeline de entrenamiento.

    Contrato:
      - execute(ctx) es el único método obligatorio.
      - __call__ es el punto de entrada real — agrega logging y trazabilidad
        automáticamente sin que cada step tenga que repetirlo.

    Ejemplo de subclase mínima:
        class MyStep(BaseTrainingStep):
            def execute(self, ctx: TrainingContext) -> TrainingContext:
                ctx.clean_data = ctx.raw_data.dropna()
                return ctx
    """

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
        """Nombre del step — por defecto el nombre de la clase."""
        return self.__class__.__name__