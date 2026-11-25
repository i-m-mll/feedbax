"""Names derived from authored keys for Studio presentation."""

import re


def member_display_name(key: str, label: str | None = None) -> str:
    return label or re.sub(r"[-_]+", " ", key).capitalize()
