class ModelNotLoadedError(RuntimeError):
    def __init__(self, name: str) -> None:
        super().__init__(f"El modelo '{name}' no fue cargado. Llamá load() primero.")


class PredictProbaNotSupportedError(NotImplementedError):
    def __init__(self, name: str) -> None:
        super().__init__(f"El modelo '{name}' no soporta predict_proba().")
        
        
        
# Excepciones del registry
class ModelNotFoundError(KeyError):
    """El modelo solicitado no existe en el registry."""
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Modelo '{name}' no encontrado en el registry.")


class ModelNotReadyError(RuntimeError):
    """El modelo existe pero no fue cargado correctamente."""
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Modelo '{name}' existe pero no está listo (is_loaded=False).")


class ModelAlreadyExistsError(ValueError):
    """Se intenta registrar un nombre que ya existe sin allow_override."""
    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(
            f"Modelo '{name}' ya está registrado. "
            "Usá allow_override=True para reemplazarlo."
        )
