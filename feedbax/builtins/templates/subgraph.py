"""Editable nested graph template."""

from pydantic import BaseModel, ConfigDict

from feedbax.contracts.portable import ComponentOptions, GraphTemplate


class _SubgraphConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Subgraph(GraphTemplate):
    """Editable nested graph container."""

    config = _SubgraphConfig
    component_options = ComponentOptions(is_composite=True)

    @classmethod
    def expand(cls, config):
        from feedbax.contracts.graph import GraphSpec

        return GraphSpec()
