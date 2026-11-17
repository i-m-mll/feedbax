"""The authoring CLI can select a registry or authored document family."""

import json

from feedbax.cli import main
from feedbax.contracts.figure_document import FIGURE_DOCUMENT_SCHEMA


def test_describe_selects_current_registry_and_document_family(capsys):
    assert main(["describe", "registry:operations"]) == 0
    operations = json.loads(capsys.readouterr().out)
    assert {"feedbax.aligned_velocity", "feedbax.summarize_velocity"} <= {
        item["key"] for item in operations["entries"]
    }
    assert main(["describe", FIGURE_DOCUMENT_SCHEMA]) == 0
    assert json.loads(capsys.readouterr().out)["schema_id"] == FIGURE_DOCUMENT_SCHEMA

    assert main(["describe", "receipt"]) == 2
    rejected = capsys.readouterr()
    assert not rejected.out
    assert "receipt" in rejected.err
    assert "step:report" in rejected.err and "family:figure" in rejected.err
