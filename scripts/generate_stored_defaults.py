"""Record the constants used to omit authored project fields."""

import json
from pathlib import Path

from feedbax.contracts.project_storage import stored_defaults_manifest


def main() -> None:
    manifest = Path(__file__).resolve().parents[1] / "feedbax/contracts/stored_defaults_contract.json"
    text = json.dumps(stored_defaults_manifest(), ensure_ascii=False, indent=2, allow_nan=False)
    manifest.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
