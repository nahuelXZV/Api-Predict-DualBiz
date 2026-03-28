from datetime import datetime
from typing import Generic, TypeVar
from dataclasses import dataclass, field

from app.domain.core.config import tz_now

T = TypeVar("T")


@dataclass
class ResponseDTO(Generic[T]):
    success: bool
    message: str
    data: T | None = None
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=tz_now)
