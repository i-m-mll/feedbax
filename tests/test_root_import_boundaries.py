import importlib

import pytest


def test_feedbax_misc_is_not_a_root_compatibility_module():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("feedbax.misc")
