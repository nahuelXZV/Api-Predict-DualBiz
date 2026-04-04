from dataclasses import dataclass


@dataclass
class SearchCVConfig:
    n_iter: int = 30
    cv: int = 3
    random_state: int = 42
