"""Filename formatting utilities."""


def format_scientific_notation(
    num: float,
    precision: int,
) -> str:
    """Format a number using compact scientific notation."""
    if precision < 0:
        raise ValueError("precision must be non-negative.")

    formatted = f"{num:.{precision}e}"
    mantissa, exponent = formatted.split("e")

    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent = str(int(exponent))

    return f"{mantissa}e{exponent}"
