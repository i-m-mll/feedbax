"""Backward compatibility shim - yaml is now in feedbax.config.yaml."""
from feedbax.config.yaml import *
from feedbax.config.yaml import (
    get_yaml_loader,
    REPRESENTERS,
    CONSTRUCTORS,
    MULTI_CONSTRUCTORS,
    _YamlLiteral,
)
