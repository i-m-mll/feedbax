import clsx from 'clsx';

interface ValueCountBadgeProps {
  count: number;
  className?: string;
}

/** Compact count for a multiple-value entry surface. */
export function ValueCountBadge({ count, className }: ValueCountBadgeProps) {
  const label = `${count} value${count === 1 ? '' : 's'}`;
  return (
    <span
      aria-label={label}
      title={label}
      className={clsx(
        'pointer-events-none inline-flex min-w-5 items-center justify-center rounded-full bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold tabular-nums text-slate-500',
        className
      )}
    >
      {count}
    </span>
  );
}
