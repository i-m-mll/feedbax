# Downstream interface stability during the rebuild

The Feedbax owner’s [retirement ruling](https://app.mandible.localhost:19420/issue/4007bf2#4e6c7d76) governs the current integration line. The pre-retirement [owner-ratified policy and exact guarantee tables](https://github.com/i-m-mll/feedbax/blob/47fe615970e7c4d320f174d16587385d95433ea1/docs/design/downstream_interface_stability.md) remain available at the last revision containing the retired implementation. That record preserves policy identity `feedbax.downstream-interface-stability.v1`, the original effective release `0.2.0`, the protocol window and component cutover exception, and the protected ratification merges `b6697280324b3a675cf1de5fbca25b42a0f56795` and `798c085268119074f0522e3a2313a1722dfaedc8`.

## Current interfaces and evidence

These are implemented interfaces on the rebuild line. The linked tests identify the behavior exercised in the repository.

| Interface | Current behavior and its checks |
| --- | --- |
| Plugin registration | The [plugin facade](../../feedbax/plugins/__init__.py) exposes typed registration and the [bootstrap](../../feedbax/plugins/bootstrap.py) checks numeric downstream protocol 2 before registering a plugin. [Bootstrap checks](../../tests/test_plugin_bootstrap.py) cover dependency order, provenance, invalid declarations and sealed registries. [Component registration checks](../../tests/test_component_registration.py) exercise the explicit owner pack and component history. |
| Project authoring and Plan | A saved project has one `<project-name>.feedbax` declaration in its folder. The [project loader](../../feedbax/contracts/project_io.py) and [compiler](../../feedbax/contracts/project_plan.py) drive the text and Studio Plan paths. [CLI Plan checks](../../tests/test_text_project_plan_cli.py) cover declaration selection and refusal of ambiguous or unsafe paths. [Request identity checks](../../tests/test_request_identity.py) cover canonical bytes and diagnostics. |
| Managed Train and Evaluate | [Jobs](../../feedbax/jobs.py) admits saved requests and the [local Host](../../feedbax/host/daemon.py) owns execution. [Train custody checks](../../tests/test_canonical_train_admission.py) and [finite Evaluate checks](../../tests/test_saved_finite_evaluate.py) exercise exact saved requests, typed checkpoints, restart and Resume, and checkpoint selection. |
| Command entry points | The [package metadata](../../pyproject.toml) declares `feedbax`, `feedbax-run`, `feedbax-analysis` and `feedbax-figure`. The [CLI dispatcher](../../feedbax/cli.py) and its delegates define the currently available commands. |
| Provider primitives | The [RunPod transport](../../feedbax/orchestration/runpod.py) and its [tests](../../tests/test_runpod_transport.py) cover the independent provider boundary. The M4 session driver owns remote job lifecycle adoption and its conformance evidence. |

## Retired lifecycle and recorded results

The owner retired controller intent/effect execution, the GCP worker, StageEngine and run-set lifecycle, the legacy HTTP worker and executor, and envelope CLI execution. The exact import and entry-point inventory and managed acceptance evidence are recorded in the [retirement handoff](https://app.mandible.localhost:19420/issue/9c27c14#92fd7452). The old policy rows and external cases describe their pinned historical runtime. This includes `public_lifecycle_recovery`, `custody_persistence_recovery`, and the result schema `feedbax.external_conformance.result.v15`. A schema identity records the shape of an old result; it supplies no current execution evidence. The [historical conformance fixture and runner](https://github.com/i-m-mll/feedbax/tree/47fe615970e7c4d320f174d16587385d95433ea1/tests/fixtures/downstream) belong to that revision.

Development projects and stores may be recreated while the rebuild changes formats. At M5, a tool outside the live runtime will read selected archived scientific outputs at their pinned revision and write them into managed projects. The [retirement ruling](https://app.mandible.localhost:19420/issue/4007bf2#4e6c7d76) defines that scope.

A new public stability commitment for a managed import, CLI command, HTTP surface or durable format requires an explicit owner decision and matching conformance evidence. The [historical policy](https://github.com/i-m-mll/feedbax/blob/47fe615970e7c4d320f174d16587385d95433ea1/docs/design/downstream_interface_stability.md) records the adopted policy and evidence for its release line.
