import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from app.domain.core.logging import logger
from app.domain.ml.pipeline_context import BaseContext

T = TypeVar("T", bound=BaseContext)


class StepABC(ABC, Generic[T]):
    @abstractmethod
    def execute(self, ctx: T) -> T: ...

    def __call__(self, ctx: T) -> T:
        logger.info("step_started", step=self.name, model=ctx.model_name)
        inicio = time.monotonic()
        error_msg = None

        try:
            ctx = self.execute(ctx)
        except Exception as e:
            error_msg = str(e)
            ctx.errors.append(error_msg)
            logger.error("step_failed", step=self.name, model=ctx.model_name, error=error_msg)
        finally:
            duracion = round(time.monotonic() - inicio, 3)
            estado = "fallido" if error_msg else "exitoso"
            ctx.steps_executed.append(self.name)

            if ctx.ejecucion_id is not None:
                self._log_step(ctx.ejecucion_id, len(ctx.steps_executed), estado, duracion, error_msg)

        logger.info("step_finished", step=self.name, model=ctx.model_name, estado=estado, duracion=duracion)
        return ctx

    def _log_step(self, ejecucion_id: int, orden: int, estado: str, duracion: float, error_msg: str | None) -> None:
        try:
            from app.application.services.log_tarea_service import log_tarea_service
            log_tarea_service.registrar_step(
                ejecucion_id=ejecucion_id,
                nombre_step=self.name,
                orden_step=orden,
                estado=estado,
                duracion_segundos=duracion,
                mensaje_error=error_msg,
            )
        except Exception as e:
            logger.warning("log_step_error", step=self.name, error=str(e))

    @property
    def name(self) -> str:
        return self.__class__.__name__
