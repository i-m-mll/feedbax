"""Host entrypoint for read-only result checks and quick environment replay."""

from collections.abc import Callable
from pathlib import Path
import os

from feedbax.contracts.receipt_validity import CheckOutcome, ValidityCheckDelivery, ValidityCheckEvidence
from feedbax.host.io import append
from feedbax.validity.recording import assert_execution_unchanged


def main(argv: list[str]) -> int:
    delivery = ValidityCheckDelivery.model_validate_json(Path(os.environ["FEEDBAX_HOST_INPUT"]).read_bytes())
    log = Path(os.environ["FEEDBAX_HOST_EVENT_LOG"])
    try:
        verify = _observation_guard(delivery)
        verify()
        evidence = _check(delivery)
        verify()
        assert_execution_unchanged()
    except (OSError, ValueError) as error:
        evidence = ValidityCheckEvidence(**{
            domain: CheckOutcome(outcome="unavailable", detail=str(error) or type(error).__name__)
            for domain in delivery.checks
        })
    append(log, "result", {"outcome": "succeeded", "validity_evidence": evidence.model_dump(mode="json")}, sync=True)
    return 0


def _observation_guard(delivery: ValidityCheckDelivery) -> Callable[[], None]:
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.effective_success import SuccessView
    from feedbax.application.composition import load_application
    from feedbax.validity.project_admission import ProjectValidity

    store = ImmutableArtifactBlobProvider(delivery.store_root)
    published = SuccessView(store).read_success(delivery.receipt)
    if published.receipt.request.id != delivery.request_id:
        raise ValueError("Validity check received data for a different request")
    registries = load_application().bundle

    def verify() -> None:
        view = SuccessView(store)
        if view.authored_ref(delivery.request_id) != delivery.receipt:
            raise ValueError("The latest successful receipt changed before the validity check")
        observed = ProjectValidity(delivery.project_root, registries).inspect(view, published)
        if observed.conditions != delivery.conditions or any(
            reason.kind in ("file", "input") for reason in observed.decision.reasons
        ):
            raise ValueError("The current replay conditions differ from those requested")

    return verify


def _check(delivery: ValidityCheckDelivery) -> ValidityCheckEvidence:
    match delivery.request_id.partition(":")[0]:
        case "train":
            from feedbax.host.training_worker import check_train

            return check_train(delivery)
        case "evaluate":
            from feedbax.host.evaluate_worker import check_evaluate

            return check_evaluate(delivery)
        case _:
            return _check_light(delivery)


def _check_light(delivery: ValidityCheckDelivery) -> ValidityCheckEvidence:
    from feedbax.host.light_worker import check_light_environment
    from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
    from feedbax.persistence.effective_success import SuccessView

    store = ImmutableArtifactBlobProvider(delivery.store_root)
    view = SuccessView(store)
    previous = view.read_success(delivery.receipt).receipt
    if previous.request.id != delivery.request_id or delivery.checks != ("environment",):
        raise ValueError("Quick replay needs the exact successful request and environment check")
    view.verify_bindings(previous.request, previous.inputs)
    matches = check_light_environment(
        store, view.read_request(previous.request), previous.inputs, previous.products,
        os.environ["FEEDBAX_HOST_JOB_ID"],
    )
    return ValidityCheckEvidence(environment=CheckOutcome(
        outcome="equal" if matches else "different",
        detail="Quick replay reproduced the retained products within tolerance." if matches
        else "The quick replay's products differ from the retained products.",
    ))
