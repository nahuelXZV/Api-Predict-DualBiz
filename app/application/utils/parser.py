def parse_bool(value, default=False):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")

    if isinstance(value, (int, float)):
        return value != 0

    return default

def parse_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default
    
def parse_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default