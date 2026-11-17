from numbers import Real
import re

import numpy as np
from _plotly_utils.basevalidators import ColorValidator


_COLOR_VALIDATOR = ColorValidator("color", "scatter")


def color_add_alpha(rgb_str: str, alpha: float) -> str:
    """Add an alpha channel to an RGB or six-digit hex color."""
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number between 0 and 1")
    alpha_value = float(alpha)
    if not np.isfinite(alpha_value) or not 0 <= alpha_value <= 1:
        raise ValueError("alpha must be a finite number between 0 and 1")
    if alpha_value == 0:
        alpha_value = 0.0
    alpha_str = np.format_float_positional(alpha_value, trim="0")

    rgb_match = re.fullmatch(r"rgb\(([^()]*)\)", rgb_str, flags=re.IGNORECASE)
    if rgb_match is not None:
        components = rgb_match.group(1).split(",")
        if len(components) == 3:
            literal = f"rgba({rgb_match.group(1)}, {alpha_str})"
            try:
                return _COLOR_VALIDATOR.validate_coerce(literal)
            except ValueError:
                pass

            normalized_components = []
            for component in components:
                stripped = component.strip()
                percentage = stripped.endswith("%")
                number_text = stripped[:-1] if percentage else stripped
                decimal_pattern = r"(?:\d+(?:\.\d*)?|\.\d+)"
                if not re.fullmatch(
                    rf"[+]?{decimal_pattern}(?:[eE][+-]?\d+)?", number_text
                ):
                    break
                plotly_safe = re.fullmatch(decimal_pattern, number_text) and all(
                    character == " " or not character.isspace() for character in component
                )
                if plotly_safe:
                    normalized_components.append(component)
                else:
                    value = float(number_text)
                    if not np.isfinite(value):
                        break
                    maximum = 100.0 if percentage else 255.0
                    normalized_number = np.format_float_positional(
                        min(value, maximum), trim="-"
                    )
                    normalized_components.append(
                        f"{normalized_number}{'%' if percentage else ''}"
                    )
            else:
                normalized = f"rgba({','.join(normalized_components)}, {alpha_str})"
                try:
                    return _COLOR_VALIDATOR.validate_coerce(normalized)
                except ValueError:
                    pass

    if re.fullmatch(r"#[0-9a-fA-F]{6}", rgb_str):
        red, green, blue = (int(rgb_str[index : index + 2], 16) for index in (1, 3, 5))
        return f"rgba({red}, {green}, {blue}, {alpha_str})"

    raise ValueError("color must use rgb(r,g,b) or #rrggbb format")
