def parse_bool(value, default=False):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")

    if isinstance(value, (int, float)):
        return value != 0

    return default
