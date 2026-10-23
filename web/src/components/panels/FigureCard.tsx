import type { FigureInfo } from '@/types/figures';

function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

function formatTime(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
}

export function FigureCard({
  figure,
  onClick,
}: {
  figure: FigureInfo;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="rounded-xl border border-slate-100 bg-slate-50/70 p-3 text-left hover:border-brand-300 hover:bg-brand-50/30 transition-colors cursor-pointer w-full"
    >
      {/* Identifier */}
      <div className="text-sm font-semibold text-slate-800 truncate" title={figure.identifier}>
        {figure.identifier}
      </div>

      {/* Figure type badge */}
      <div className="mt-1.5 flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full border border-brand-200 text-brand-600 bg-brand-500/10">
          {figure.figure_type}
        </span>
        {figure.pert__type && (
          <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full border border-slate-200 text-slate-500">
            {figure.pert__type}
            {figure.pert__std != null && ` (${figure.pert__std})`}
          </span>
        )}
      </div>

      {/* Experiment name */}
      {figure.expt_name && (
        <div className="mt-1.5 text-[10px] text-slate-400 truncate" title={figure.expt_name}>
          {figure.expt_name}
        </div>
      )}

      {/* Date and formats */}
      <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400">
        <span>{formatDate(figure.created_at)} {formatTime(figure.created_at)}</span>
        <span className="tabular-nums">{figure.saved_formats.join(', ')}</span>
      </div>
    </button>
  );
}
