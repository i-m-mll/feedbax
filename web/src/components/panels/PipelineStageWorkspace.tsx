import { Boxes, FileJson, GitBranch, Link2, PackageCheck } from 'lucide-react';
import { stageProductReferences } from '@/features/scenario/integration';
import type {
  StudioArtifactRef,
  StudioCollectionRef,
  StudioManifestRef,
  StudioScenarioSpec,
  StudioStageSpec,
  StudioWorkspaceSpec,
} from '@/types/workspace';

function formatCount(count: number, singular: string, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function refLabel(ref: StudioManifestRef | StudioArtifactRef): string {
  return ref.role ?? ref.kind;
}

function RefList({
  title,
  refs,
}: {
  title: string;
  refs: Array<StudioManifestRef | StudioArtifactRef>;
}) {
  return (
    <section className="space-y-2">
      <div className="text-[10px] uppercase tracking-[0.22em] text-slate-400">{title}</div>
      {refs.length === 0 ? (
        <div className="text-xs text-slate-400">None recorded</div>
      ) : (
        <div className="space-y-1.5">
          {refs.map((ref) => (
            <div key={`${ref.kind}:${ref.id}`} className="border-t border-slate-100 pt-1.5">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="font-medium text-slate-700 truncate">{refLabel(ref)}</span>
                <span className="text-[10px] text-slate-400">{ref.provider}</span>
              </div>
              <div className="mt-0.5 truncate text-[11px] text-slate-400">{ref.id}</div>
              {ref.uri && <div className="mt-0.5 truncate text-[11px] text-slate-400">{ref.uri}</div>}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function CollectionList({
  title,
  collections,
}: {
  title: string;
  collections: StudioCollectionRef[];
}) {
  return (
    <section className="space-y-2">
      <div className="text-[10px] uppercase tracking-[0.22em] text-slate-400">{title}</div>
      {collections.length === 0 ? (
        <div className="text-xs text-slate-400">None selected</div>
      ) : (
        <div className="space-y-2">
          {collections.map((collection) => (
            <div key={collection.id} className="border-t border-slate-100 pt-2">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="font-medium text-slate-700 truncate">
                  {collection.label ?? collection.kind}
                </span>
                <span className="text-[10px] text-slate-400">
                  {formatCount(collection.item_refs.length, 'item')}
                </span>
              </div>
              <div className="mt-0.5 truncate text-[11px] text-slate-400">{collection.id}</div>
              {collection.item_refs.slice(0, 3).map((ref) => (
                <div key={`${collection.id}:${ref.id}`} className="mt-1 truncate text-[11px] text-slate-500">
                  {ref.role ?? ref.kind}: {ref.id}
                </div>
              ))}
              {collection.item_refs.length > 3 && (
                <div className="mt-1 text-[11px] text-slate-400">
                  +{collection.item_refs.length - 3} more
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export function StageProvenancePanel({
  stage,
  scenario,
  workspace,
}: {
  stage: StudioStageSpec | null;
  scenario: StudioScenarioSpec | null;
  workspace: StudioWorkspaceSpec | null;
}) {
  if (!stage) {
    return (
      <aside className="hidden w-80 shrink-0 border-l border-slate-100 bg-white/70 p-4 text-sm text-slate-400 lg:block">
        No active stage
      </aside>
    );
  }
  const references = stageProductReferences(workspace, stage.id);

  return (
    <aside className="hidden w-80 shrink-0 overflow-y-auto border-l border-slate-100 bg-white/70 p-4 lg:block">
      <div className="space-y-5">
        <section>
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em] text-slate-400">
            <GitBranch className="h-3.5 w-3.5" />
            Stage
          </div>
          <div className="mt-1 text-sm font-semibold text-slate-800">{stage.label}</div>
          <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
            <span className="rounded-full bg-slate-100 px-2 py-0.5">{stage.kind}</span>
            <span className="rounded-full bg-slate-100 px-2 py-0.5">{stage.status}</span>
          </div>
          {scenario && (
            <div className="mt-2 text-xs text-slate-500">
              Scenario: <span className="font-medium text-slate-700">{scenario.label}</span>
            </div>
          )}
          {typeof stage.metadata.draft_version === 'number' && (
            <div className="mt-1 text-[11px] text-slate-400">
              Draft version {stage.metadata.draft_version}
            </div>
          )}
        </section>

        <CollectionList title="Inputs" collections={stage.input_collections} />
        <CollectionList title="Outputs" collections={stage.output_collections} />
        <StageReferenceList references={references} />
        <RefList title="Manifests" refs={stage.manifest_refs} />
        <RefList title="Artifacts" refs={stage.artifact_refs ?? []} />
      </div>
    </aside>
  );
}

function StageReferenceList({
  references,
}: {
  references: ReturnType<typeof stageProductReferences>;
}) {
  const meaningful = references.filter(
    (reference) =>
      reference.kind === 'analysis_page' ||
      reference.kind === 'report_section' ||
      reference.itemCount > 0
  );
  return (
    <section className="space-y-2">
      <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em] text-slate-400">
        <Link2 className="h-3.5 w-3.5" />
        Stage Links
      </div>
      {meaningful.length === 0 ? (
        <div className="text-xs text-slate-400">No products linked</div>
      ) : (
        <div className="space-y-2">
          {meaningful.map((reference) => (
            <div key={reference.id} className="border-t border-slate-100 pt-2">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="truncate font-medium text-slate-700">{reference.label}</span>
                <span className="shrink-0 rounded bg-slate-100 px-1.5 py-0.5 text-[10px] text-slate-500">
                  {reference.kind.replace('_', ' ')}
                </span>
              </div>
              <div className="mt-0.5 truncate text-[11px] text-slate-400">
                {reference.collectionId ?? reference.summary ?? reference.stageKind}
              </div>
              {reference.manifestIds.slice(0, 2).map((id) => (
                <div key={id} className="mt-1 truncate text-[11px] text-slate-500">
                  {id}
                </div>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

export function StageDraftPanel({
  stage,
  scenario,
}: {
  stage: StudioStageSpec | null;
  scenario: StudioScenarioSpec | null;
}) {
  if (!stage) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-400">
        No active pipeline stage
      </div>
    );
  }

  const hasSelection = Object.keys(stage.selection_spec).length > 0;
  const hasExecutionSpec = stage.execution_spec && Object.keys(stage.execution_spec).length > 0;

  return (
    <div className="h-full overflow-y-auto p-6 text-sm text-slate-600">
      <div className="max-w-3xl space-y-6">
        <div>
          <div className="text-xs uppercase tracking-[0.3em] text-slate-400">{stage.kind}</div>
          <div className="mt-1 text-lg font-semibold text-slate-800">{stage.label}</div>
          {scenario && <div className="mt-1 text-xs text-slate-500">{scenario.label}</div>}
        </div>

        <section className="space-y-2 border-t border-slate-100 pt-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
            <Boxes className="h-4 w-4" />
            Collections
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <div className="text-xs font-medium text-slate-700">Inputs</div>
              <div className="mt-1 text-xs text-slate-500">
                {formatCount(
                  stage.input_collections.reduce((total, collection) => total + collection.item_refs.length, 0),
                  'item'
                )}
              </div>
            </div>
            <div>
              <div className="text-xs font-medium text-slate-700">Outputs</div>
              <div className="mt-1 text-xs text-slate-500">
                {formatCount(
                  stage.output_collections.reduce((total, collection) => total + collection.item_refs.length, 0),
                  'item'
                )}
              </div>
            </div>
          </div>
        </section>

        <section className="space-y-2 border-t border-slate-100 pt-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
            <PackageCheck className="h-4 w-4" />
            Selection
          </div>
          {hasSelection ? (
            <pre className="max-h-48 overflow-auto rounded border border-slate-100 bg-slate-50 p-3 text-xs text-slate-600">
              {JSON.stringify(stage.selection_spec, null, 2)}
            </pre>
          ) : (
            <div className="text-xs text-slate-400">No selection spec yet</div>
          )}
        </section>

        <section className="space-y-2 border-t border-slate-100 pt-4">
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-slate-400">
            <FileJson className="h-4 w-4" />
            Execution Draft
          </div>
          {hasExecutionSpec ? (
            <pre className="max-h-56 overflow-auto rounded border border-slate-100 bg-slate-50 p-3 text-xs text-slate-600">
              {JSON.stringify(stage.execution_spec, null, 2)}
            </pre>
          ) : (
            <div className="text-xs text-slate-400">No execution spec prepared</div>
          )}
        </section>
      </div>
    </div>
  );
}
