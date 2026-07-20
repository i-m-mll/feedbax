# Governed golden path

The governed golden path is one provider-free, local scenario: a content- and
implementation-digest-pinned v3 authored row is migrated to the current v5
matrix schema, lowered by an installed plugin, assembled with an authenticated
checkpoint-custody input, and run through the public `shadow-launch` entrypoint.
The same installed plugin must provide both the row lowerer and the
descriptor-backed execution preparation; the launch may not inject
`--initial-slots`.

Success means a fresh process materializes the pinned custody archive, restores
the prepared topology, and records exactly one continuation batch (cumulative
total two). The fixture reads the emitted checkpoint `latest.json`, follows its
transaction-manifest path, and verifies that manifest's parent lineage against
the seeded parent transaction. It persists one bundle and uses only a local
immutable artifact provider.

That exact persisted bundle must then be accepted in a separate fresh process
by the shared RunPod `dry_run_launch_bundle` input-binding validation from
[issue:14834f5]. The fixture makes the RunPod transport and network surfaces
raise if reached. Acceptance is symmetric up to provider availability: the
local one-update rehearsal and driver validation accept the same persisted
bundle and its input declarations/native execution bindings. The driver check
stops before acquisition, provisioning, monitoring, retries, credentials, or
any provider/network operation. This fixture owns only this acceptance seam;
the shadow entrypoint belongs to [issue:8fcfe12] and the driver implementation
belongs to [issue:14834f5].
