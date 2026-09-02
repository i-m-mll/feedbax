#!/usr/bin/env python3
"""Generate and check Feedbax's durable identity map.

The map is intentionally derived from declarations in this file plus the live
source tree.  Python anchors are fingerprinted from normalized ASTs and the
Studio TypeScript anchors are fingerprinted from their function bodies.  A
hash-bearing class field added anywhere under ``feedbax/`` changes the generated
inventory, so the checked-in artifacts cannot remain green without review.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Literal


ROOT = Path(__file__).resolve().parents[1]
MAP_PATH = ROOT / "docs/design/durable_identity_map.md"
INVENTORY_PATH = ROOT / "docs/design/durable_identity_field_inventory.v1.json"


@dataclass(frozen=True)
class Anchor:
    path: str
    symbol: str
    language: Literal["python", "typescript"] = "python"


@dataclass(frozen=True)
class Encoder:
    id: str
    identity_kind: str
    algorithm: str
    version: str
    bytes_contract: str
    producers: tuple[Anchor, ...]


@dataclass(frozen=True)
class Surface:
    id: str
    meaning: str
    identity_kind: str
    encoders: tuple[str, ...]
    carriers: tuple[str, ...]
    assertions: tuple[Anchor, ...]
    downstream: str
    boundary: str


@dataclass(frozen=True)
class CanonicalJsonMigrationDecision:
    carriers: str
    current_domain: str
    decision: str
    version_boundary: str


ENCODERS = (
    Encoder(
        id="canonical-json-v1",
        identity_kind="semantic content identity",
        algorithm="SHA-256",
        version="canonical_json_v1",
        bytes_contract=(
            "Parsed JSON or a null-omitted Pydantic JSON projection; sorted keys, compact "
            "separators, UTF-8. Python json defaults remain part of this v1 domain: ASCII "
            "escaping is enabled and non-finite floats are not rejected by the encoder."
        ),
        producers=(
            Anchor("feedbax/contracts/canonical_json.py", "canonical_json_v1_bytes"),
            Anchor("feedbax/contracts/manifest.py", "canonical_json_bytes"),
            Anchor("feedbax/contracts/authored_canonical.py", "canonical_sha256"),
        ),
    ),
    Encoder(
        id="canonical-json-v2",
        identity_kind="strict cross-language semantic JSON identity",
        algorithm="SHA-256",
        version="canonical_json_v2",
        bytes_contract=(
            "Strict JSON encoded as compact UTF-8 with unnormalized non-ASCII text, deterministic "
            "JSON escapes, UTF-16 code-unit object-key order, ECMAScript-compatible shortest "
            "finite binary64 number spelling, negative zero normalized to zero, and integers "
            "restricted to the JavaScript safe range. Non-JSON values, lone surrogates, cycles, "
            "non-string keys, NaN, infinities, and unsafe integers reject with typed errors."
        ),
        producers=(
            Anchor("feedbax/contracts/canonical_json.py", "canonical_json_v2_bytes"),
            Anchor("feedbax/contracts/canonical_json.py", "canonical_json_bytes_for_algorithm"),
        ),
    ),
    Encoder(
        id="training-spec-json-current",
        identity_kind="semantic intent and resolved execution identity",
        algorithm="SHA-256",
        version="unversioned encoder; carried by versioned schema families",
        bytes_contract=(
            "Normalized JSON with string-only object keys, sorted compact UTF-8, non-ASCII "
            "characters emitted directly, non-finite floats rejected, and negative zero "
            "normalized to zero."
        ),
        producers=(
            Anchor("feedbax/contracts/spec_storage.py", "training_spec_canonical_bytes"),
            Anchor("feedbax/contracts/spec_storage.py", "training_spec_sha256"),
        ),
    ),
    Encoder(
        id="publication-json-v1",
        identity_kind="semantic protocol-record identity",
        algorithm="SHA-256",
        version="feedbax.publication.v1",
        bytes_contract=(
            "Strict protocol JSON model projection; sorted compact UTF-8, non-ASCII characters "
            "emitted directly, and non-finite floats rejected. Negative zero is not normalized."
        ),
        producers=(Anchor("feedbax/contracts/publication.py", "canonical_bytes"),),
    ),
    Encoder(
        id="raw-sha256",
        identity_kind="byte/material identity",
        algorithm="SHA-256",
        version="raw bytes",
        bytes_contract=(
            "The exact byte string read, written, archived, transferred, or held in custody; "
            "no JSON parsing or canonicalization is implied."
        ),
        producers=(
            Anchor("feedbax/contracts/manifest.py", "sha256_bytes"),
            Anchor("feedbax/contracts/manifest.py", "sha256_file"),
            Anchor("feedbax/contracts/publication.py", "BlobRef.from_bytes"),
        ),
    ),
    Encoder(
        id="checkpoint-content-v2",
        identity_kind="mixed semantic structure and exact checkpoint material identity",
        algorithm="SHA-256",
        version="feedbax.training_checkpoint.structural_abi.content.v2",
        bytes_contract=(
            "Array leaves use contiguous C-order bytes. Structural records, leaf lists, slot "
            "roots, and transaction roots use canonical-json-v1; the run-contract projection "
            "additionally normalizes signed zero."
        ),
        producers=(
            Anchor("feedbax/training/checkpoint_custody.py", "structural_abi_fingerprint"),
            Anchor("feedbax/training/checkpoint_custody.py", "_leaf_content_digests"),
            Anchor("feedbax/training/checkpoint_custody.py", "_slot_root_sha256"),
            Anchor("feedbax/training/checkpoint_custody.py", "_transaction_root_sha256"),
            Anchor("feedbax/training/checkpoint_custody.py", "_run_contract_hash"),
        ),
    ),
    Encoder(
        id="evaluation-states-v3",
        identity_kind="mixed array/material and metadata identity",
        algorithm="SHA-256",
        version="feedbax.manifest.evaluation_states_container.v3",
        bytes_contract=(
            "Array leaves use contiguous C-order bytes; metadata leaves use sorted compact JSON "
            "with non-finite floats rejected; the structure fingerprint names the PyTree shape."
        ),
        producers=(
            Anchor("feedbax/contracts/evaluation_states.py", "_array_digest"),
            Anchor("feedbax/contracts/evaluation_states.py", "_canonical_json_bytes"),
            Anchor("feedbax/contracts/evaluation_states.py", "_treedef_structure_fingerprint"),
        ),
    ),
    Encoder(
        id="value-identity-v1",
        identity_kind="authored, semantic numeric, and runtime realization identities",
        algorithm="SHA-256",
        version="feedbax.value_identity.v1",
        bytes_contract=(
            "Authored and realization envelopes use training-spec-json-current. Semantic values use "
            "a versioned JSON header followed by little-endian C-order array bytes with signed "
            "zero and NaN normalization."
        ),
        producers=(
            Anchor("feedbax/contracts/value_identity.py", "authored_value_sha256"),
            Anchor("feedbax/contracts/value_identity.py", "semantic_value_sha256"),
            Anchor("feedbax/contracts/value_identity.py", "realization_value_sha256"),
        ),
    ),
    Encoder(
        id="studio-fnv1a-draft",
        identity_kind="persisted presentation-revision comparison",
        algorithm="32-bit FNV-1a",
        version="unversioned fnv1a prefix contract",
        bytes_contract=(
            "A recursively key-sorted JSON-like string. Python and TypeScript implementations "
            "are separate current producers; integral floats and negative zero are known to "
            "serialize differently and are owned by the Studio convergence lanes."
        ),
        producers=(
            Anchor("feedbax/studio/execution.py", "_stable_ui_hash"),
            Anchor(
                "web/src/utils/pipelineCollections.ts",
                "stableHash",
                language="typescript",
            ),
        ),
    ),
    Encoder(
        id="legacy-model-md5-v2",
        identity_kind="serialized model material identity",
        algorithm="MD5",
        version="ModelRecord.hash_version v2",
        bytes_contract=(
            "The YAML hyperparameter prefix and Equinox serialized leaves, byte for byte. The "
            "version field is mandatory and other versions fail closed."
        ),
        producers=(Anchor("feedbax/persistence/database.py", "hash_pytree"),),
    ),
    Encoder(
        id="legacy-evaluation-md5",
        identity_kind="legacy semantic record identity",
        algorithm="MD5",
        version="unversioned legacy evaluation/figure contract",
        bytes_contract=(
            "UTF-8 of joined model hashes, names, and default-spaced sorted JSON parameters; "
            "figure identity adds the evaluation hash and figure identifier."
        ),
        producers=(
            Anchor("feedbax/persistence/database.py", "generate_eval_hash"),
            Anchor("feedbax/persistence/database.py", "generate_figure_hash"),
        ),
    ),
    Encoder(
        id="implementation-json-sha256",
        identity_kind="producer implementation provenance",
        algorithm="SHA-256",
        version="producer-specific declared dependency projection",
        bytes_contract=(
            "A sorted compact JSON projection of the declared implementation identity and its "
            "registered dependencies. It identifies producing code, not authored scientific JSON."
        ),
        producers=(
            Anchor(
                "feedbax/training/row_lowering.py",
                "training_row_lowerer_implementation_sha256",
            ),
            Anchor(
                "feedbax/training/authoring.py",
                "training_method_authoring_implementation_sha256",
            ),
        ),
    ),
    Encoder(
        id="repo-realization-sha256",
        identity_kind="environment and source-material realization identity",
        algorithm="SHA-256",
        version="feedbax orchestration current schemas",
        bytes_contract=(
            "Schema-specific sorted compact JSON projections plus exact lockfile, patch, staged "
            "root, and repository snapshot byte digests. These identities describe a realized "
            "execution environment, not authored experiment semantics."
        ),
        producers=(
            Anchor("feedbax/orchestration/repo_snapshot.py", "snapshot_manifest_digest"),
            Anchor("feedbax/orchestration/repo_realization.py", "repo_realization_plan_digest"),
            Anchor("feedbax/orchestration/drivers/local.py", "compute_environment_fingerprint"),
        ),
    ),
)


SURFACES = (
    Surface(
        id="experiment-compile-lock",
        meaning="Pins authored inputs, compiled output, and the pre-run execution identity.",
        identity_kind="semantic content identity plus quoted material receipts",
        encoders=("canonical-json-v1", "raw-sha256"),
        carriers=(
            "ExperimentCompileLock envelope.envelope_hash + pin_algorithm",
            "base/lineage/content_pin content_hash + pin_algorithm",
            "PlannedProductReference envelope_hash + compiled_content_hash",
            "RowProvenanceReference source_content_hash + pin_algorithm",
            "AuthenticatedReceiptReference manifest_sha256 + size_bytes",
            "compiled_document.content_hash + pin_algorithm",
            "execution_identity.sha256 + ordered inputs + pin_algorithm",
        ),
        assertions=(
            Anchor("feedbax/contracts/experiment_compile_lock.py", "build_compile_lock"),
            Anchor("feedbax/contracts/experiment_compile_lock.py", "load_compile_lock"),
            Anchor(
                "tests/test_envelope_engine_kernel.py",
                "test_execution_identity_is_stable_for_identical_inputs",
            ),
        ),
        downstream=(
            "rlrmp2 generated/*.compile-lock.json and its authored envelopes; the pinned snapshot "
            "contains 26 compile locks, all carrying canonical_json_v1 pins and execution_identity."
        ),
        boundary=(
            "All content_hash fields are canonical parsed-document identity. manifest_sha256 is "
            "an exact receipt-file digest and must not be described as canonical JSON."
        ),
    ),
    Surface(
        id="worker-consistency-predicate",
        meaning="Pins the phase program from which checkpoint consistency rules were derived.",
        identity_kind="semantic content identity",
        encoders=("canonical-json-v1", "canonical-json-v2"),
        carriers=(
            "ConsistencyPredicateSpec.phase_program_digest + pin_algorithm",
        ),
        assertions=(
            Anchor(
                "feedbax/contracts/worker.py",
                "migrate_consistency_predicate_payload",
            ),
            Anchor(
                "tests/test_canonical_json.py",
                "test_consistency_predicate_v2_migration_preserves_and_pins_its_digest",
            ),
            Anchor(
                "tests/test_canonical_json.py",
                "test_new_consistency_predicates_pin_canonical_json_v2",
            ),
        ),
        downstream="Checkpoint transactions embed ConsistencyPredicateSpec as worker authority.",
        boundary=(
            "Migrated v2 records retain v1 digest meaning; newly derived v3 records use v2. The "
            "pin selects the verifier and unknown pins reject."
        ),
    ),
    Surface(
        id="compiler-graph-and-workspace-anchor",
        meaning="Binds an authored graph document to resolved graph semantics and Studio state.",
        identity_kind="semantic content identity",
        encoders=("canonical-json-v1",),
        carriers=(
            "DocumentRoot.content_sha256",
            "ResolvedGraph.document_sha256 + resolved_sha256",
            "CompilationRecord/CompilationFailureRecord document_sha256 (+ resolved_sha256)",
            "SemanticAnchor.semantic_document_sha256",
        ),
        assertions=(
            Anchor("feedbax/compiler/graph.py", "ResolvedGraph.validate_identity"),
            Anchor(
                "tests/test_graph_compiler.py",
                "test_compile_graph_is_deterministic_and_records_runtime_key_order",
            ),
        ),
        downstream="Studio WorkspaceDocument semantic_root and semantic_anchors.",
        boundary="These SHA-256 values are not the Studio fnv1a spec_hashes used for stale badges.",
    ),
    Surface(
        id="manifest-spec-and-composition",
        meaning="Carries semantic spec identities, composition layers, and content-pinned parents.",
        identity_kind="semantic content identity with exact artifact references",
        encoders=("canonical-json-v1", "raw-sha256"),
        carriers=(
            "SpecPayload.sha256 + source_sha256",
            "ContentPinnedJsonBase.sha256",
            "CanonicalJsonDocumentPin.sha256",
            "analysis/evaluation/training/figure composition envelope_sha256 + parent_sha256",
            "FigureCompositionDocument sha256 + selected_sha256",
            "row index index_sha256 and expression hashes/results digest",
            "TrainingRunManifest intent_hash + execution_hash + resolved_semantics_root_hash",
        ),
        assertions=(
            Anchor("feedbax/contracts/manifest.py", "_ensure_spec_payload_hash"),
            Anchor("feedbax/contracts/matrix_core.py", "load_content_pinned_json_base"),
            Anchor("feedbax/contracts/figures.py", "figure_composition_payload_identity_sha256"),
        ),
        downstream=(
            "rlrmp2 imports canonical_json_bytes directly for comparator, stabilization, and "
            "post-run identities and keeps content-pinned specs plus generated composition outputs."
        ),
        boundary=(
            "SpecPayload.source_sha256 and artifact/manifest refs can name source or stored bytes; "
            "their producer must be followed before assigning a canonical JSON domain."
        ),
    ),
    Surface(
        id="training-spec-storage",
        meaning="Separates authored intent, composed intent, resolved semantics, and execution.",
        identity_kind="semantic intent and execution identity",
        encoders=("training-spec-json-current", "raw-sha256"),
        carriers=(
            "TrainingSpecStorageResult intent_hash + authored_envelope_hash + composed_intent_hash",
            "TrainingSpecStorageResult resolved_root_hash + execution_hash",
            "TrainingRunExecutionCapsule dependency_lock_digest + environment_digest",
            "TrainingRunExecutionCapsule intent_hash + resolved_root_hash + execution_hash",
            "TrainingRunMatrix row payload/authored/lowered/runtime hashes and resolved/execution hashes",
            "MaterialDependencyAdmission.identity_sha256",
        ),
        assertions=(
            Anchor(
                "feedbax/contracts/spec_storage.py",
                "TrainingRunExecutionCapsule._validate_identity",
            ),
            Anchor("feedbax/contracts/spec_storage.py", "training_run_execution_hash"),
            Anchor(
                "tests/test_training_spec_storage.py",
                "test_snapshot_rows_are_complete_and_seed_changes_execution_identity",
            ),
        ),
        downstream=(
            "rlrmp2 production code imports training_spec_sha256 for adaptive-lambda row lowering, "
            "fork locks, preparation, and materialized task identity."
        ),
        boundary=(
            "dependency_lock_digest and environment_digest are material/environment inputs quoted "
            "inside a semantic capsule; they are not authored JSON identities."
        ),
    ),
    Surface(
        id="publication-and-logical-artifact",
        meaning="Joins exact blobs to logical artifact, checkpoint, and publication identities.",
        identity_kind="semantic protocol identity over exact material references",
        encoders=("publication-json-v1", "raw-sha256"),
        carriers=(
            "BlobRef.digest + size_bytes",
            "ExactRef.bytes",
            "ArtifactRecord.version_id",
            "CheckpointSet.checkpoint_id and exact_ref.bytes",
            "PublicationRequest.request_sha256 + publication_id",
            "PublicationReceipt.request_sha256 + exact refs",
        ),
        assertions=(
            Anchor("feedbax/contracts/publication.py", "ArtifactRecord._validate_record"),
            Anchor("feedbax/contracts/publication.py", "CheckpointSet._validate_checkpoint"),
            Anchor(
                "tests/test_publication_protocol.py",
                "test_publication_is_idempotent_and_conflicting_replay_fails_closed",
            ),
        ),
        downstream="rlrmp2's native CheckpointSet and publication-adoption acceptance path.",
        boundary="BlobRef.digest always identifies exact bytes, even when those bytes contain JSON.",
    ),
    Surface(
        id="checkpoint-transaction-and-fork",
        meaning="Authenticates checkpoint leaves, slots, transaction roots, run contracts, and forks.",
        identity_kind="semantic structure plus exact serialized material",
        encoders=("checkpoint-content-v2", "raw-sha256", "implementation-json-sha256"),
        carriers=(
            "StructuralAbiFingerprint fingerprint_algorithm_version + fingerprint_sha256",
            "SlotLeafContentDigest.sha256",
            "SlotContentDigest blob_sha256 + leaf_hashes + slot_root_sha256",
            "ContentIntegrityDigest.transaction_root_sha256",
            "RunContractBinding hash_domain and its *_sha256 fields",
            "CheckpointForkSourceRecord manifest_sha256 + transaction_root_sha256",
            "CheckpointForkCompatibilityProjection run_contract_projection_sha256 + slot_structural_abi_sha256",
            "CheckpointTransactionManifest content_integrity_digest and slot refs",
            "checkpoint archive manifest/transaction/archive SHA-256 evidence",
        ),
        assertions=(
            Anchor("feedbax/training/checkpoint_custody.py", "load_checkpoint_set"),
            Anchor("feedbax/training/checkpoint_custody.py", "load_checkpoint_custody_documents"),
            Anchor(
                "tests/test_checkpoint_custody.py",
                "test_checkpoint_transaction_derives_slots_and_loads_multi_slot_state",
            ),
        ),
        downstream=(
            "rlrmp2 pinned checkpoint-root, archive, manifest, slot, and transform identities in "
            "fork declarations, generated locks, and release acceptance."
        ),
        boundary=(
            "Archive/manifest/blob hashes are material identity. Structural ABI, run-contract, "
            "slot-root, and transaction-root hashes are derived semantic/integrity identities."
        ),
    ),
    Surface(
        id="evaluation-state-container",
        meaning="Authenticates evaluation arrays, metadata, structure, and the stored container.",
        identity_kind="semantic structure plus exact array/container material",
        encoders=("evaluation-states-v3", "raw-sha256"),
        carriers=(
            "EvaluationStatesArrayRecord.sha256",
            "EvaluationStatesLeafRecord.sha256",
            "EvaluationStatesContainerPayloadV2/V3.metadata_sha256",
            "EvaluationStatesContainerPayloadV3.structure_fingerprint",
            "ArtifactRef.sha256 for the emitted NPZ container",
        ),
        assertions=(
            Anchor(
                "feedbax/contracts/evaluation_states.py", "load_evaluation_states_container_bytes"
            ),
            Anchor(
                "tests/test_evaluation_states_v3.py",
                "test_v3_round_trip_is_deterministic_and_preserves_namedtuple_types",
            ),
        ),
        downstream=(
            "rlrmp2 comparator and post-run products assert state-array, state-container, and "
            "manifest identities; raw channel arrays are hashed as raw C-order bytes."
        ),
        boundary="A state array sha256 is raw array bytes, not canonical JSON.",
    ),
    Surface(
        id="value-identity",
        meaning="Keeps authored declaration, normalized numeric value, and runtime realization distinct.",
        identity_kind="three explicit semantic/realization tiers",
        encoders=("value-identity-v1",),
        carriers=(
            "ValueIdentityRecord.authored_sha256",
            "ValueIdentityRecord.semantic_sha256 + expected_semantic_sha256",
            "ValueIdentityRecord.realization_sha256",
            "runtime_layout_fingerprint + runtime_backend_fingerprint",
            "authored_identity_chain",
        ),
        assertions=(
            Anchor(
                "feedbax/contracts/value_identity.py",
                "ValueIdentityRecord._validate_identity_chain_and_expectation",
            ),
            Anchor(
                "tests/test_value_identity.py",
                "test_expected_semantic_mismatch_fails_closed_and_chain_is_preserved",
            ),
        ),
        downstream="No pinned rlrmp2 artifact was found to embed feedbax.value_identity.v1 at the snapshot.",
        boundary="semantic_sha256 includes normalized numeric bytes; it is neither source JSON nor stored artifact bytes.",
    ),
    Surface(
        id="studio-spec-hashes",
        meaning="Persists lightweight stale-badge comparisons for inline Studio spec payloads.",
        identity_kind="presentation revision identity",
        encoders=("studio-fnv1a-draft",),
        carriers=(
            "StudioManifestRef.metadata.spec_hashes / snapshot_spec_hashes",
            "RunCollectionStagePanel current-vs-snapshot spec hash comparisons",
        ),
        assertions=(
            Anchor(
                "web/src/utils/pipelineCollections.ts", "stableStringify", language="typescript"
            ),
            Anchor("web/src/utils/pipelineCollections.test.ts", "describe", language="typescript"),
        ),
        downstream="Studio frontend only; no known rlrmp2 persisted consumer.",
        boundary=(
            "fnv1a values are not SHA-256 and do not authenticate model, manifest, or artifact "
            "bytes. Cross-language number spelling is a known current mismatch."
        ),
    ),
    Surface(
        id="legacy-database-identities",
        meaning="Names saved model files, evaluation records, and figure records in the legacy database.",
        identity_kind="versioned model material plus unversioned legacy semantic identities",
        encoders=("legacy-model-md5-v2", "legacy-evaluation-md5"),
        carriers=(
            "ModelRecord.hash + hash_version",
            "EvaluationRecord.hash + model_hashes",
            "FigureRecord.hash + evaluation_hash + model_hashes",
        ),
        assertions=(
            Anchor("feedbax/persistence/database.py", "validate_model_hash_version"),
            Anchor(
                "tests/test_persistence_imports.py",
                "test_model_record_rejects_unsupported_hash_version",
            ),
        ),
        downstream="Legacy Feedbax database and figure paths; no rlrmp2 source import found.",
        boundary="These MD5 identities are not any Feedbax SHA-256 canonical JSON domain.",
    ),
    Surface(
        id="implementation-provenance",
        meaning="Pins the exact registered producer/lowerer or transform implementation.",
        identity_kind="producer provenance",
        encoders=("implementation-json-sha256",),
        carriers=(
            "TrainingRowLowererRef/Registration.implementation_sha256",
            "DurableSlotTransform and CheckpointForkTransformRecord.implementation_sha256",
            "governed constructor_fingerprint inputs",
        ),
        assertions=(
            Anchor("feedbax/training/row_lowering.py", "TrainingRowLowererRegistry.lower"),
            Anchor(
                "tests/test_training_row_lowering.py",
                "test_registry_rejects_conflicting_registration_implementation",
            ),
        ),
        downstream="rlrmp2 registers and pins adaptive-lambda row-lowerer and fork-transform implementations.",
        boundary="This proves producing code identity; it does not identify authored or emitted document bytes.",
    ),
    Surface(
        id="orchestration-realization",
        meaning="Pins source snapshots, dependency locks, staged roots, bundles, and realized environments.",
        identity_kind="material and environment realization identity",
        encoders=("repo-realization-sha256", "raw-sha256"),
        carriers=(
            "RepoSnapshotRecord.content_sha256",
            "RepoRealizationEntry/Plan sealed_lock_digests + plan_digest",
            "StagedRootFileRecord.sha256 + StagedRootCustody.content_sha256",
            "EnvironmentDeclaration.lockfile_hashes",
            "RunSetState/RealizedDeploymentRecord.environment_fingerprint",
            "RunBundle and nested preflight/certificate/payload SHA-256 fields",
        ),
        assertions=(
            Anchor("feedbax/orchestration/repo_snapshot.py", "seal_repo_snapshot"),
            Anchor("feedbax/orchestration/repo_realization.py", "seal_local_repo_realizations"),
            Anchor(
                "tests/test_repo_snapshot.py",
                "test_snapshot_is_immutable_after_seal_and_distinguishes_dirty_bytes",
            ),
        ),
        downstream="rlrmp2 execution plans and release evidence quote Feedbax bundle and environment identities.",
        boundary="These fields describe deployed/source material and environment state, not experiment intent.",
    ),
)


CANONICAL_JSON_MIGRATION_DECISIONS = (
    CanonicalJsonMigrationDecision(
        carriers=(
            "ExperimentCompileLock envelope/base/lineage/content_pin/compiled_document/"
            "execution_identity digests; authored composition and manifest content pins"
        ),
        current_domain="canonical-json-v1",
        decision="remain v1; existing and newly emitted compile-lock v4 pins stay offline-verifiable",
        version_boundary="no schema migration in this lane",
    ),
    CanonicalJsonMigrationDecision(
        carriers=(
            "descriptor_basis_hash, acausal interior_content_hash, checkpoint structural JSON, "
            "run-bundle identity, staged-root identity, repo-realization plan identity, stage-input "
            "identity, preparation identity, and implementation-dependency identities"
        ),
        current_domain="canonical-json-v1",
        decision="remain byte-for-byte v1 through the shared legacy helper",
        version_boundary="byte-neutral consolidation; stored schema versions do not change",
    ),
    CanonicalJsonMigrationDecision(
        carriers="ConsistencyPredicateSpec.phase_program_digest",
        current_domain="canonical-json-v2 for newly produced records",
        decision=(
            "v2 records migrate to v3 with canonical_json_v1 pinned beside the unchanged digest; "
            "new v3 records pin canonical_json_v2"
        ),
        version_boundary="feedbax.manifest.worker.consistency_predicate.v2 -> v3",
    ),
    CanonicalJsonMigrationDecision(
        carriers="Penzai DomainCompileReport.interior_content_hash",
        current_domain="legacy default=str JSON",
        decision="remain on the legacy permissive domain; no digest is relabelled as v2",
        version_boundary="requires a future DomainCompileReport schema migration owned with Studio",
    ),
    CanonicalJsonMigrationDecision(
        carriers="expression_hash and resolved-semantics snapshot node hashes",
        current_domain="UTF-8-direct strict JSON variants",
        decision="remain on their existing domains; no digest is relabelled as v2",
        version_boundary="requires migrations of the owning expression and snapshot schemas",
    ),
    CanonicalJsonMigrationDecision(
        carriers="evaluation-states metadata and structure digests",
        current_domain="evaluation-states-v3",
        decision="remain on evaluation-states-v3; no digest is relabelled as v2",
        version_boundary="requires a future evaluation-states container schema migration",
    ),
    CanonicalJsonMigrationDecision(
        carriers="training-spec storage and publication protocol hashes",
        current_domain="training-spec-json-current and publication-json-v1",
        decision="remain separate named domains; similar strictness does not imply equal bytes",
        version_boundary="their owning durable schema families do not migrate in this lane",
    ),
    CanonicalJsonMigrationDecision(
        carriers="Studio Python/TypeScript fnv1a presentation revision hashes",
        current_domain="studio-fnv1a-draft",
        decision="remain unchanged for the dedicated Studio persistence and parity lanes",
        version_boundary="explicitly out of scope here",
    ),
)


DOWNSTREAM = {
    "repository": "github.com/i-m-mll/rlrmp-star",
    "revision": "964b9c3d31f092d2c3126915de30aa371a987e87",
    "source_anchors": {
        "canonical-json-v1": {
            "src/rlrmp2/adaptive_lambda/preparation.py": "canonical_sha256",
            "src/rlrmp2/adaptive_lambda/method_payloads.py": "canonical_sha256",
            "src/rlrmp2/comparators/contracts.py": "canonical_json_bytes",
            "src/rlrmp2/comparators/trial_materialization.py": "canonical_json_bytes",
            "src/rlrmp2/analysis/stabilization.py": "canonical_json_bytes",
            "src/rlrmp2/post_run/terminal_adversary_energy.py": "canonical_json_bytes",
        },
        "training-spec-json-current": {
            "src/rlrmp2/adaptive_lambda/mapped_fork_lock.py": "training_spec_sha256",
            "src/rlrmp2/adaptive_lambda/method.py": "training_spec_sha256",
            "src/rlrmp2/adaptive_lambda/fork.py": "training_spec_sha256",
        },
        "downstream-authored-raw-sha256": {
            "src/rlrmp2/comparators/trial_bank.py": "hashlib.sha256(values.tobytes",
            "src/rlrmp2/comparators/trial_materialization.py": "hashlib.sha256(selected.tobytes",
            "src/rlrmp2/post_run/adaptive_diagnostics_recipe.py": "hashlib.sha256(raw).hexdigest",
            "src/rlrmp2/post_run/retained_report_panels.py": "hashlib.sha256(training_input.manifest_input.raw_bytes)",
            "src/rlrmp2/post_run/terminal_adversary_energy.py": "hashlib.sha256(raw_checkpoint_manifest)",
        },
    },
    "artifact_counts": {
        "generated_compile_locks": 26,
        "compile_locks_with_execution_identity": 26,
        "compile_locks_with_canonical_json_v1": 26,
        "spec_documents_with_hash_or_pin_fields": 104,
        "result_documents_with_hash_or_pin_fields": 19,
    },
    "representative_artifacts": (
        "generated/sisu-full-configuration-release.compile-lock.json",
        "generated/mapped-fork-continuation.compile-lock.json",
        "specs/experiment/learning-rate-rewarm-continuation.envelope.json",
        "results/5a0ef7e/tier-a-downstream-acceptance.v1.json",
    ),
}


_FIELD_TERM = re.compile(
    r"(?:^hash$|sha256|(?:^|_)hash(?:es)?$|(?:^|_)digest(?:s)?$|"
    r"(?:^|_)fingerprint(?:s)?$|(?:^|_)pin(?:s)?$|hash_domain|hash_version|pin_algorithm)"
)
_NAME_COLLISIONS = (
    "damping",
    "mapping",
    "bookkeeping",
    "early_stopping",
    "mandible_manifest_mappings",
)


def _annotation_text(annotation: ast.expr) -> str:
    return ast.unparse(annotation)


def _field_domain_hint(path: str, class_name: str, field: str) -> str:
    lowered = field.lower()
    if any(part in lowered for part in _NAME_COLLISIONS):
        return "not-an-identity-name-collision"
    if path.startswith("feedbax/persistence/"):
        return "legacy-persistence"
    if "value_identity.py" in path:
        return "value-identity-v1"
    if "evaluation_states.py" in path:
        return "evaluation-states-v3"
    if path == "feedbax/contracts/worker.py" and field in {
        "phase_program_digest",
        "pin_algorithm",
    }:
        return "canonical-json-v2-or-migrated-canonical-json-v1"
    if "publication.py" in path:
        return "publication-json-v1-or-raw-sha256"
    if "checkpoints.py" in path or "checkpoint_custody.py" in path:
        return "checkpoint-content-v2-or-raw-sha256"
    if any(
        token in path
        for token in (
            "spec_storage.py",
            "run_matrix.py",
            "run_composition.py",
            "material_dependencies.py",
            "training/preparation.py",
            "training/row_lowering.py",
            "training/authoring.py",
        )
    ):
        return "training-spec-json-current-or-quoted-material"
    if any(
        token in path
        for token in (
            "repo_snapshot.py",
            "repo_realization.py",
            "staged_root_custody.py",
            "input_materialization.py",
            "orchestration/bundle.py",
            "manifest_packet.py",
        )
    ):
        return "repo-realization-sha256-or-raw-sha256"
    if path.startswith("feedbax/web/") or path.startswith("feedbax/studio/"):
        return "presentation-or-api-reference"
    if path.startswith("feedbax/testing/"):
        return "test-evidence-reference"
    if any(token in path for token in ("experiment_compile_lock.py", "compiler/graph.py")):
        return "canonical-json-v1-or-quoted-material"
    if "artifact_schema.py" in path:
        return "raw-sha256"
    if "manifest.py" in path and class_name in {
        "ArtifactRef",
        "ArrayStoreRef",
        "FileHashRef",
        "TreeHashEntry",
        "TreeHashRef",
        "AuthenticatedManifestDigest",
    }:
        return "raw-sha256"
    return "canonical-json-v1-or-cross-boundary-reference"


def discover_field_inventory() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted((ROOT / "feedbax").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for statement in node.body:
                if not isinstance(statement, ast.AnnAssign) or not isinstance(
                    statement.target, ast.Name
                ):
                    continue
                field = statement.target.id
                if not _FIELD_TERM.search(field.lower()):
                    continue
                records.append(
                    {
                        "path": relative,
                        "class": node.name,
                        "field": field,
                        "annotation": _annotation_text(statement.annotation),
                        "domain_hint": _field_domain_hint(relative, node.name, field),
                    }
                )
    return records


def inventory_document() -> dict[str, object]:
    fields = discover_field_inventory()
    return {
        "schema_id": "feedbax.governance.durable_identity_field_inventory",
        "schema_version": "feedbax.governance.durable_identity_field_inventory.v1",
        "scope": (
            "Lexical inventory of hash, digest, fingerprint, and pin-bearing annotated class "
            "fields under feedbax/. The durable identity map supplies the authoritative domain "
            "boundaries; domain_hint is a review routing aid."
        ),
        "field_count": len(fields),
        "fields": fields,
    }


def _find_python_symbol(path: Path, symbol: str) -> ast.AST:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    candidates: Iterable[ast.AST] = tree.body
    current: ast.AST | None = None
    for part in symbol.split("."):
        current = next(
            (
                node
                for node in candidates
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == part
            ),
            None,
        )
        if current is None:
            raise ValueError(f"missing Python anchor {path.relative_to(ROOT)}:{symbol}")
        candidates = getattr(current, "body", ())
    assert current is not None
    return current


def _typescript_symbol_text(path: Path, symbol: str) -> str:
    text = path.read_text(encoding="utf-8")
    patterns = (
        rf"(?:export\s+)?function\s+{re.escape(symbol)}\s*\(",
        rf"(?:export\s+)?(?:const|let)\s+{re.escape(symbol)}\s*=",
        r"(?:export\s+)?(?:describe|it|test)\s*\(",
    )
    match = next(
        (re.search(pattern, text) for pattern in patterns if re.search(pattern, text)), None
    )
    if match is None:
        raise ValueError(f"missing TypeScript anchor {path.relative_to(ROOT)}:{symbol}")
    opening = text.find("{", match.start())
    if opening < 0:
        raise ValueError(f"TypeScript anchor has no body {path.relative_to(ROOT)}:{symbol}")
    depth = 0
    quote: str | None = None
    escaped = False
    for index in range(opening, len(text)):
        character = text[index]
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character in ("'", '"', "`"):
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[match.start() : index + 1]
    raise ValueError(f"unterminated TypeScript anchor {path.relative_to(ROOT)}:{symbol}")


def anchor_fingerprint(anchor: Anchor) -> str:
    path = ROOT / anchor.path
    if not path.is_file():
        raise ValueError(f"missing anchor file {anchor.path}")
    if anchor.language == "python":
        payload = ast.dump(
            _find_python_symbol(path, anchor.symbol),
            annotate_fields=True,
            include_attributes=False,
        ).encode("utf-8")
    else:
        payload = _typescript_symbol_text(path, anchor.symbol).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _anchor_label(anchor: Anchor) -> str:
    return f"`{anchor.path}:{anchor.symbol}` (`{anchor_fingerprint(anchor)[:16]}`)"


def render_map(inventory: dict[str, object]) -> str:
    lines = [
        "# Durable identity map",
        "",
        "> Generated by `scripts/check_durable_identity_map.py`; edit the declarations in that "
        "script, regenerate, and review every changed source fingerprint. Do not hand-edit this file.",
        "",
        "This is the current Feedbax contract. It maps an authored or material input to the exact "
        "encoder that produces a durable identity, the record that carries it, the assertion that "
        "checks it, and the known downstream dependency. It does not make incompatible byte domains "
        "equivalent.",
        "",
        "A **semantic identity** answers whether two declared meanings are the same after the stated "
        "projection. A **byte/material identity** answers whether exact stored or transferred bytes "
        "are the same. A digest can be 64 lowercase hexadecimal characters in both cases without "
        "belonging to the same domain.",
        "",
        "## Encoders",
        "",
        "| Encoder | Kind | Algorithm / version | Authoritative bytes | Producer provenance |",
        "| --- | --- | --- | --- | --- |",
    ]
    for encoder in ENCODERS:
        producers = "<br>".join(_anchor_label(anchor) for anchor in encoder.producers)
        lines.append(
            f"| `{encoder.id}` | {encoder.identity_kind} | {encoder.algorithm}; "
            f"`{encoder.version}` | {encoder.bytes_contract} | {producers} |"
        )

    lines.extend(
        [
            "",
            "## Canonical JSON migration decisions",
            "",
            "The shared `canonical_json_v2` conformance vector is "
            "`conformance/canonical_json_v2.json`. Python consumes it now; TypeScript consumers "
            "must consume the same tracked file rather than restating its cases.",
            "",
            "| Durable carriers | Current domain | Decision | Version boundary |",
            "| --- | --- | --- | --- |",
        ]
    )
    for decision in CANONICAL_JSON_MIGRATION_DECISIONS:
        lines.append(
            f"| {decision.carriers} | `{decision.current_domain}` | {decision.decision} | "
            f"{decision.version_boundary} |"
        )

    lines.extend(["", "## Durable surfaces", ""])
    for surface in SURFACES:
        lines.extend(
            [
                f"### {surface.id}",
                "",
                surface.meaning,
                "",
                f"- Identity kind: {surface.identity_kind}.",
                f"- Encoder(s): {', '.join(f'`{item}`' for item in surface.encoders)}.",
                "- Carriers:",
                "",
                *[f"  - `{carrier}`" for carrier in surface.carriers],
                "",
                "- In-repo assertions:",
                "",
                *[f"  - {_anchor_label(anchor)}" for anchor in surface.assertions],
                "",
                f"- Known downstream: {surface.downstream}",
                f"- Boundary: {surface.boundary}",
                "",
            ]
        )

    counts = DOWNSTREAM["artifact_counts"]
    lines.extend(
        [
            "## Pinned downstream snapshot",
            "",
            f"Repository `{DOWNSTREAM['repository']}` at commit `{DOWNSTREAM['revision']}` was read "
            "without modification. This revision is evidence provenance, not a promise that its "
            "moving branch still has the same contents.",
            "",
            "Direct producer dependencies:",
            "",
        ]
    )
    source_anchors = DOWNSTREAM["source_anchors"]
    assert isinstance(source_anchors, dict)
    for domain, entries in source_anchors.items():
        lines.append(f"- `{domain}`:")
        assert isinstance(entries, dict)
        for path, needle in entries.items():
            lines.append(f"  - `{path}` (`{needle}`)")
    lines.extend(
        [
            "",
            "Pinned corpus counts:",
            "",
            *[f"- `{key}`: {value}" for key, value in counts.items()],
            "",
            "Representative pinned artifacts:",
            "",
            *[f"- `{path}`" for path in DOWNSTREAM["representative_artifacts"]],
            "",
            "The `downstream-authored-raw-sha256` rows are deliberately separate. Those producers "
            "hash raw arrays, manifests, or artifact bytes in rlrmp2. They are not Feedbax "
            "canonical JSON merely because the field is named `sha256`.",
            "",
            "## Drift guard",
            "",
            f"The companion inventory currently contains {inventory['field_count']} annotated "
            "hash/digest/fingerprint/pin field candidates from `feedbax/`. It records every exact "
            "class field and a domain-routing hint in "
            "`docs/design/durable_identity_field_inventory.v1.json`.",
            "",
            "`uv run --no-sync python scripts/check_durable_identity_map.py --check` fails when:",
            "",
            "- a mapped producer or assertion symbol disappears or its normalized implementation changes;",
            "- a hash-bearing annotated class field is added, removed, renamed, or changes type;",
            "- either checked-in generated artifact differs from the current source; or",
            "- an optional pinned-downstream verification finds different source anchors or corpus counts.",
            "",
            "To verify the pinned downstream snapshot from a local checkout, run "
            "`uv run --no-sync python scripts/check_durable_identity_map.py --check "
            "--downstream-root <rlrmp2-checkout>`. "
            "The verifier reads the pinned Git tree with optional locks disabled; it does not inspect "
            "or modify downstream working-tree bytes.",
            "",
            "## Current non-equivalences",
            "",
            "- `canonical-json-v1`, `canonical-json-v2`, `training-spec-json-current`, and "
            "`publication-json-v1` use different JSON byte contracts. A refactor may share code only "
            "after proving the emitted bytes and schema migration boundary appropriate to each "
            "stored field.",
            "- Worker consistency-predicate v2 records migrate to v3 by pinning their unchanged "
            "`phase_program_digest` to `canonical_json_v1`; only newly derived v3 records use and pin "
            "`canonical_json_v2`.",
            "- Compile-lock `execution_identity.sha256` is pinned to canonical-json-v1 in v4. The "
            "explicit v3 migration asserts that pin only for locks with attributable built-in "
            "Feedbax compiler provenance; downstream-authored and unattributed digest shapes remain "
            "outside that migration.",
            "- Studio `fnv1a:` values are presentation revision markers, not authenticity proofs, and "
            "the current Python/TypeScript number spelling is not one byte domain.",
            "- Raw manifest, artifact, archive, array, and checkpoint blob digests remain exact material "
            "identity even when the bytes happen to decode as JSON.",
            "- Legacy persistence MD5 fields remain their declared current contracts and are never "
            "silently reinterpreted as SHA-256.",
            "",
        ]
    )
    return "\n".join(lines)


def render_inventory(document: dict[str, object]) -> str:
    return json.dumps(document, indent=2, sort_keys=False) + "\n"


def _git(repo: Path, *arguments: str) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        env=environment,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return result.stdout


def _git_text(repo: Path, revision: str, path: str) -> str:
    return _git(repo, "show", f"{revision}:{path}")


def verify_downstream(repo: Path) -> None:
    revision = str(DOWNSTREAM["revision"])
    _git(repo, "cat-file", "-e", f"{revision}^{{commit}}")
    source_anchors = DOWNSTREAM["source_anchors"]
    assert isinstance(source_anchors, dict)
    for entries in source_anchors.values():
        assert isinstance(entries, dict)
        for path, needle in entries.items():
            if needle not in _git_text(repo, revision, path):
                raise ValueError(f"pinned downstream anchor drifted: {path}: {needle}")
    names = _git(repo, "ls-tree", "-r", "--name-only", revision).splitlines()
    compile_locks = [
        name
        for name in names
        if name.startswith("generated/") and name.endswith(".compile-lock.json")
    ]
    compile_texts = [_git_text(repo, revision, path) for path in compile_locks]
    specs = [name for name in names if name.startswith("specs/") and name.endswith(".json")]
    results = [name for name in names if name.startswith("results/") and name.endswith(".json")]

    def contains_hash_field(text: str) -> bool:
        return any(
            token in text
            for token in (
                '"sha256"',
                '"content_hash"',
                '"manifest_sha256"',
                '"checkpoint_root_hash"',
            )
        )

    observed = {
        "generated_compile_locks": len(compile_locks),
        "compile_locks_with_execution_identity": sum(
            '"execution_identity"' in text for text in compile_texts
        ),
        "compile_locks_with_canonical_json_v1": sum(
            "canonical_json_v1" in text for text in compile_texts
        ),
        "spec_documents_with_hash_or_pin_fields": sum(
            contains_hash_field(_git_text(repo, revision, path)) for path in specs
        ),
        "result_documents_with_hash_or_pin_fields": sum(
            contains_hash_field(_git_text(repo, revision, path)) for path in results
        ),
    }
    if observed != DOWNSTREAM["artifact_counts"]:
        raise ValueError(
            "pinned downstream corpus counts drifted: "
            f"expected={DOWNSTREAM['artifact_counts']!r}, observed={observed!r}"
        )
    for path in DOWNSTREAM["representative_artifacts"]:
        _git(repo, "cat-file", "-e", f"{revision}:{path}")


def _check_file(path: Path, expected: str) -> bool:
    if not path.is_file():
        print(f"missing generated artifact: {path.relative_to(ROOT)}", file=sys.stderr)
        return False
    actual = path.read_text(encoding="utf-8")
    if actual != expected:
        print(
            f"stale generated artifact: {path.relative_to(ROOT)}; "
            "run scripts/check_durable_identity_map.py",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="check instead of writing")
    parser.add_argument(
        "--downstream-root",
        type=Path,
        help="optionally verify the pinned rlrmp2 Git tree",
    )
    args = parser.parse_args()

    inventory = inventory_document()
    rendered_inventory = render_inventory(inventory)
    rendered_map = render_map(inventory)
    if args.downstream_root is not None:
        verify_downstream(args.downstream_root.resolve())
    if args.check:
        ok = _check_file(INVENTORY_PATH, rendered_inventory)
        ok = _check_file(MAP_PATH, rendered_map) and ok
        return 0 if ok else 1
    INVENTORY_PATH.write_text(rendered_inventory, encoding="utf-8")
    MAP_PATH.write_text(rendered_map, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
