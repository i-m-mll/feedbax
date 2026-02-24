import { useMemo } from 'react';
import { CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';
import clsx from 'clsx';
import type { DiagnosticsResponse, DiagnosticCheck } from '@/types/statistics';

const STATUS_CONFIG = {
  pass: {
    icon: CheckCircle2,
    iconColor: 'text-emerald-500',
    bg: 'bg-emerald-50',
    border: 'border-emerald-100',
    label: 'Pass',
  },
  warn: {
    icon: AlertTriangle,
    iconColor: 'text-amber-500',
    bg: 'bg-amber-50',
    border: 'border-amber-100',
    label: 'Warning',
  },
  fail: {
    icon: XCircle,
    iconColor: 'text-red-500',
    bg: 'bg-red-50',
    border: 'border-red-100',
    label: 'Fail',
  },
} as const;

function DiagnosticRow({ check }: { check: DiagnosticCheck }) {
  const config = STATUS_CONFIG[check.status];
  const Icon = config.icon;

  return (
    <div
      className={clsx(
        'rounded-lg border p-3 flex items-start gap-3',
        config.bg,
        config.border,
      )}
    >
      <Icon className={clsx('w-4 h-4 mt-0.5 flex-shrink-0', config.iconColor)} />
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium text-slate-700">{check.name}</div>
        <div className="text-xs text-slate-500 mt-0.5">{check.reason}</div>
        {check.hint && (
          <div className="text-xs text-slate-400 mt-1 italic">{check.hint}</div>
        )}
      </div>
    </div>
  );
}

export function DiagnosticsPanel({ data }: { data: DiagnosticsResponse }) {
  const counts = useMemo(() => {
    const c = { pass: 0, warn: 0, fail: 0 };
    for (const check of data.checks) {
      c[check.status]++;
    }
    return c;
  }, [data]);

  if (!data.checks.length) {
    return <div className="text-sm text-slate-400 p-4">No diagnostics available</div>;
  }

  return (
    <div className="space-y-3">
      {/* Summary bar */}
      <div className="flex items-center gap-4 text-xs font-semibold">
        {counts.pass > 0 && (
          <span className="flex items-center gap-1 text-emerald-600">
            <CheckCircle2 className="w-3.5 h-3.5" />
            {counts.pass} passed
          </span>
        )}
        {counts.warn > 0 && (
          <span className="flex items-center gap-1 text-amber-600">
            <AlertTriangle className="w-3.5 h-3.5" />
            {counts.warn} warnings
          </span>
        )}
        {counts.fail > 0 && (
          <span className="flex items-center gap-1 text-red-600">
            <XCircle className="w-3.5 h-3.5" />
            {counts.fail} failed
          </span>
        )}
      </div>

      {/* Check list — failures first, then warnings, then passes */}
      <div className="space-y-2">
        {data.checks
          .slice()
          .sort((a, b) => {
            const order = { fail: 0, warn: 1, pass: 2 };
            return order[a.status] - order[b.status];
          })
          .map((check, i) => (
            <DiagnosticRow key={`${check.name}-${i}`} check={check} />
          ))}
      </div>
    </div>
  );
}
