from __future__ import annotations

import pytest

from feedbax.config.tree import _expand_missing_levels

pytestmark = [pytest.mark.feedbax_contract, pytest.mark.no_silent_substitution_contract]


def test_missing_ldict_level_without_reference_keys_rejected() -> None:
    with pytest.raises(ValueError, match="Cannot expand missing LDict level"):
        _expand_missing_levels(
            {"leaf": 1},
            target_levels=["condition"],
            current_levels=[],
            reference_trees=None,
        )
