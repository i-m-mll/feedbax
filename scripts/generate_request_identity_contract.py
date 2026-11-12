"""Generate the absent-value contract and check it against a previous revision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from feedbax.contracts.request_identity import generate_identity_fields
from feedbax.contracts.request_identity_guard import (
    validate_identity_contract,
    validate_identity_transition,
)

_CONTRACT = Path("feedbax/contracts/request_identity_contract.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Compare against the contract at this Git revision")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if args.write and not args.base:
        parser.error("--write requires --base to compare existing absences and retirements")
    checked = json.loads(_CONTRACT.read_text())
    candidate = {**checked, "fields": generate_identity_fields()}
    validate_identity_contract(candidate)
    if args.base:
        old = subprocess.check_output(
            ["git", "show", f"{args.base}:{_CONTRACT.as_posix()}"], timeout=15
        )
        validate_identity_transition(json.loads(old), candidate)
    if args.write:
        _CONTRACT.write_text(json.dumps(candidate, sort_keys=True, indent=2) + "\n")
    elif checked != candidate:
        raise SystemExit("Request identity declarations disagree with the checked contract")


if __name__ == "__main__":
    main()
