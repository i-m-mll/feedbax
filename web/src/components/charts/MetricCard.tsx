import type { MetricSummary } from '@/types/statistics';

function fmt(value: number): string {
  if (Math.abs(value) >= 100) return value.toFixed(1);
  if (Math.abs(value) >= 1) return value.toFixed(2);
  if (Math.abs(value) >= 0.01) return value.toFixed(3);
  return value.toExponential(2);
}

export function MetricCard({
  label,
  summary,
}: {
  label: string;
  summary: MetricSummary;
}) {
  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-3 min-w-[140px]">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 mb-1">
        {label}
      </div>
      <div className="text-lg font-bold text-slate-800 tabular-nums">
        {fmt(summary.median)}
      </div>
      <div className="text-[10px] text-slate-400 tabular-nums mt-0.5">
        Q25&ndash;Q75: {fmt(summary.q25)}&ndash;{fmt(summary.q75)}
      </div>
      <div className="flex items-center gap-2 text-[10px] text-slate-400 mt-0.5 tabular-nums">
        <span>&mu; {fmt(summary.mean)}</span>
        <span>&sigma; {fmt(summary.std)}</span>
      </div>
      <div className="text-[10px] text-slate-300 tabular-nums mt-0.5">
        n={summary.count}
      </div>
    </div>
  );
}
