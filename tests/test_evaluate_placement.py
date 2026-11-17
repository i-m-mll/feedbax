"""Test placement of checkpoint-linked Evaluate work."""

from feedbax.training.validation_placement import ValidationBudget, place_validation


def test_placement_respects_readiness_and_full_product_budget():
    budget = ValidationBudget(durable_point_p95_ms=4, steady_update_ms=10, max_point_ms=10, max_update_fraction=0.10)

    def place(updates=(7, 14), **facts):
        return place_validation(updates, **({
            "producer_started": False, "live_products_supported": True, "budget": budget,
        } | facts))

    assert place().location == "live"
    assert place((1, 7)).reason == "over_budget"
    assert place(producer_started=True).reason == "late"
    assert place(live_products_supported=False).reason == "unqualified"
    assert place(budget=None).reason == "unmeasured"
