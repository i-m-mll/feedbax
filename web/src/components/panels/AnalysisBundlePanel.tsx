import { useCallback, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { AlertTriangle, CheckCircle2, CircleDashed, Link2, RefreshCcw } from 'lucide-react';
import { dryRunAnalysisBundle } from '@/api/analysisAPI';
import { apiErrorMessage } from '@/api/request';
import { getActiveStage, getScenario, useWorkspaceStore } from '@/stores/workspaceStore';
import type { AnalysisBundleDryRunResult, ManifestPredicate } from '@/generated/studioContracts';
import {
  analysisBundleCards,
  bundleWithPredicate,
  selectionSpecForAnalysisStage,
  selectionSpecWithPredicate,
  stageReason,
  statusLabel,
  type AnalysisBundleCard,
  type AnalysisBundleStageStatus,
} from '@/utils/analysisBundle';

function statusTone(status: AnalysisBundleStageStatus | string): string {
  switch (status) {
    case 'would_run':
      return 'border-emerald-200 bg-emerald-50 text-emerald-700';
    case 'would_skip':
      return 'border-amber-200 bg-amber-50 text-amber-700';
    case 'not_applicable':
      return 'border-slate-200 bg-slate-100 text-slate-600';
    case 'missing':
      return 'border-red-200 bg-red-50 text-red-700';
    default:
      return 'border-slate-200 bg-white text-slate-600';
  }
}

function StatusChip({
  status,
  reason,
}: {
  status: AnalysisBundleStageStatus | string;
  reason?: string | null;
}) {
  return (
    <span
      title={reason ?? statusLabel(status)}
      className={clsx(
        'inline-flex h-6 items-center rounded-full border px-2 text-[11px] font-semibold',
        statusTone(status)
      )}
    >
      {statusLabel(status)}
    </span>
  );
}

function PredicateEditor({
  predicate,
  onCommit,
}: {
  predicate: ManifestPredicate;
  onCommit: (predicate: ManifestPredicate) => void;
}) {
  const [draft, setDraft] = useState(() => JSON.stringify(predicate, null, 2));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(JSON.stringify(predicate, null, 2));
    setError(null);
  }, [predicate]);

  const commit = useCallback(() => {
    try {
      const parsed = JSON.parse(draft) as ManifestPredicate;
      onCommit(parsed);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Invalid predicate JSON');
    }
  }, [draft, onCommit]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-400">
          Predicate
        </div>
        <button
          type="button"
          onClick={commit}
          className="inline-flex h-7 items-center gap-1.5 rounded-md border border-slate-200 bg-white px-2 text-[11px] font-semibold text-slate-600 hover:bg-slate-50"
        >
          <RefreshCcw className="h-3.5 w-3.5" />
          Retarget
        </button>
      </div>
      <textarea
        value={draft}
        onChange={(event) => setDraft(event.target.value)}
        className="h-28 w-full resize-none rounded-md border border-slate-200 bg-slate-50 p-2 font-mono text-[11px] leading-4 text-slate-700 outline-none focus:border-brand-300 focus:bg-white"
        spellCheck={false}
      />
      {error && (
        <div className="flex items-center gap-1.5 text-[11px] text-red-600">
          <AlertTriangle className="h-3.5 w-3.5" />
          {error}
        </div>
      )}
    </div>
  );
}

function BundleCard({
  card,
  dryRun,
  busy,
  error,
  onRetarget,
}: {
  card: AnalysisBundleCard;
  dryRun: AnalysisBundleDryRunResult | null;
  busy: boolean;
  error: string | null;
  onRetarget: (predicate: ManifestPredicate) => void;
}) {
  const stages = dryRun?.stages ?? [];
  const firstBlocking = stages.find(
    (stage) => stage.status === 'missing' || stage.status === 'would_skip'
  );

  return (
    <section className="grid min-w-0 gap-4 rounded-md border border-slate-200 bg-white p-4 shadow-sm lg:grid-cols-[minmax(16rem,22rem)_minmax(0,1fr)]">
      <div className="min-w-0 space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-slate-800">{card.title}</div>
            <div className="mt-1 text-xs text-slate-500">
              {card.stageCount} stages / {card.pageCount} pages
            </div>
          </div>
          {firstBlocking ? (
            <StatusChip status={firstBlocking.status} reason={stageReason(firstBlocking)} />
          ) : (
            <StatusChip status={stages.length > 0 ? 'would_run' : 'missing'} />
          )}
        </div>
        {card.description && <div className="text-xs text-slate-500">{card.description}</div>}
        <PredicateEditor predicate={card.bundle.predicate} onCommit={onRetarget} />
      </div>

      <div className="min-w-0 space-y-3">
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <span className="inline-flex items-center gap-1.5">
            <Link2 className="h-3.5 w-3.5 text-slate-400" />
            {dryRun ? `${dryRun.match_preview.match_count} matches` : 'No preview'}
          </span>
          {dryRun?.match_preview.truncated && <span>Preview truncated</span>}
          {busy && <span>Refreshing</span>}
        </div>
        {error && (
          <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
            {error}
          </div>
        )}
        {dryRun && dryRun.match_preview.parent_refs.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {dryRun.match_preview.parent_refs.slice(0, 6).map((ref) => (
              <span
                key={`${ref.kind}:${ref.id}`}
                title={ref.id}
                className="max-w-56 truncate rounded bg-slate-100 px-2 py-1 text-[11px] text-slate-600"
              >
                {ref.role ?? ref.kind}: {ref.id}
              </span>
            ))}
          </div>
        )}
        <div className="grid gap-2 md:grid-cols-2">
          {stages.map((stage) => {
            const reason = stageReason(stage);
            const outputs = stage.outputs ?? [];
            return (
              <div key={stage.name} className="min-w-0 rounded-md border border-slate-100 p-3">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0 truncate text-xs font-semibold text-slate-700">
                    {stage.name}
                  </div>
                  <StatusChip status={stage.status} reason={reason} />
                </div>
                <div className="mt-1 text-[11px] text-slate-400">{stage.kind}</div>
                {outputs.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1.5">
                    {outputs.map((output) => (
                      <StatusChip
                        key={output.role}
                        status={output.status}
                        reason={output.reason}
                      />
                    ))}
                  </div>
                )}
                {reason && (
                  <div className="mt-2 max-h-8 overflow-hidden text-[11px] text-slate-500">
                    {reason}
                  </div>
                )}
              </div>
            );
          })}
          {!dryRun && !busy && !error && (
            <div className="rounded-md border border-dashed border-slate-200 p-3 text-xs text-slate-400">
              Dry-run pending
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export function AnalysisBundlePanel() {
  const workspace = useWorkspaceStore((state) => state.workspace);
  const updateStageDraft = useWorkspaceStore((state) => state.updateStageDraft);
  const updateScenarioDraft = useWorkspaceStore((state) => state.updateScenarioDraft);
  const activeStage = getActiveStage(workspace);
  const activeScenario = getScenario(workspace, activeStage?.scenario_id);
  const selectionSpec = useMemo(
    () => selectionSpecForAnalysisStage(activeStage),
    [activeStage]
  );
  const cards = useMemo(
    () => analysisBundleCards(activeScenario, activeStage),
    [activeScenario, activeStage]
  );
  const [dryRuns, setDryRuns] = useState<Record<string, AnalysisBundleDryRunResult>>({});
  const [busyIds, setBusyIds] = useState<Set<string>>(() => new Set());
  const [errors, setErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    let cancelled = false;
    if (!activeStage || activeStage.kind !== 'analysis') return;
    cards.forEach((card) => {
      setBusyIds((current) => new Set(current).add(card.id));
      dryRunAnalysisBundle({
        bundle: card.bundle,
        selectionSpec,
      })
        .then((dryRun) => {
          if (cancelled) return;
          setDryRuns((current) => ({ ...current, [card.id]: dryRun }));
          setErrors((current) => {
            const next = { ...current };
            delete next[card.id];
            return next;
          });
        })
        .catch((error) => {
          if (cancelled) return;
          setErrors((current) => ({
            ...current,
            [card.id]: apiErrorMessage(error, 'Bundle dry-run failed'),
          }));
        })
        .finally(() => {
          if (cancelled) return;
          setBusyIds((current) => {
            const next = new Set(current);
            next.delete(card.id);
            return next;
          });
        });
    });
    return () => {
      cancelled = true;
    };
  }, [activeStage, cards, selectionSpec]);

  const retarget = useCallback(
    (card: AnalysisBundleCard, predicate: ManifestPredicate) => {
      if (!activeStage) return;
      const nextSelection = selectionSpecWithPredicate(predicate);
      updateStageDraft(
        activeStage.id,
        {
          selection_spec: nextSelection as unknown as Record<string, unknown>,
        },
        'analysis_bundle_predicate_retargeted'
      );
      if (activeScenario) {
        const currentSpec =
          activeScenario.analysis_spec &&
          typeof activeScenario.analysis_spec === 'object' &&
          !Array.isArray(activeScenario.analysis_spec)
            ? activeScenario.analysis_spec
            : {};
        updateScenarioDraft(
          activeScenario.id,
          {
            analysis_spec: {
              ...currentSpec,
              bundle: bundleWithPredicate(card.bundle, predicate),
            },
          },
          'analysis_bundle_predicate_retargeted'
        );
      }
    },
    [activeScenario, activeStage, updateScenarioDraft, updateStageDraft]
  );

  return (
    <div className="border-b border-slate-100 bg-slate-50/70 px-4 py-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">
          <CircleDashed className="h-4 w-4" />
          Analysis bundle
        </div>
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />
          Dry run only
        </div>
      </div>
      <div className="space-y-3">
        {cards.map((card) => (
          <BundleCard
            key={card.id}
            card={card}
            dryRun={dryRuns[card.id] ?? null}
            busy={busyIds.has(card.id)}
            error={errors[card.id] ?? null}
            onRetarget={(predicate) => retarget(card, predicate)}
          />
        ))}
      </div>
    </div>
  );
}
