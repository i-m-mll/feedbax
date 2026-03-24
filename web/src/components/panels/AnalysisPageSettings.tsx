/**
 * AnalysisPageSettings — right-sidebar content shown when the analysis tab
 * is active and no node is selected.
 *
 * Contains per-page settings, including the eval run selector.
 */

import { useAnalysisStore } from '@/stores/analysisStore';
import { EvalRunSelector } from '@/components/panels/RunSelector';

export function AnalysisPageSettings() {
  const activePageId = useAnalysisStore((s) => s.activePageId);
  const pages = useAnalysisStore((s) => s.pages);
  const evalRunId = useAnalysisStore((s) => s.evalRunId);
  const setEvalRunId = useAnalysisStore((s) => s.setEvalRunId);

  const activePage = pages.find((p) => p.id === activePageId);

  if (!activePage) {
    return (
      <div className="p-4 text-xs text-slate-400">
        No analysis page selected.
      </div>
    );
  }

  return (
    <div className="p-4 space-y-5">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">
          Page Settings
        </div>
        <div className="mt-1 text-sm font-medium text-slate-700">
          {activePage.name}
        </div>
      </div>

      {/* Eval run selector — per-page */}
      <div className="space-y-1.5">
        <div className="text-[10px] uppercase tracking-[0.2em] text-slate-400">
          Evaluation Run
        </div>
        <EvalRunSelector
          selectedEvalRunId={evalRunId}
          onSelectEvalRun={setEvalRunId}
        />
        <div className="text-[10px] text-slate-400 leading-relaxed">
          Select which evaluation run to use for analyses on this page.
        </div>
      </div>
    </div>
  );
}
