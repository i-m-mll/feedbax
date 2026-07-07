import clsx from 'clsx';
import { ChevronDown, X } from 'lucide-react';
import type { ReactNode } from 'react';

interface PanelSectionHeaderProps {
  title: string;
  expanded: boolean;
  onToggle: () => void;
  badges?: ReactNode;
  className?: string;
}

export function PanelSectionHeader({
  title,
  expanded,
  onToggle,
  badges,
  className,
}: PanelSectionHeaderProps) {
  return (
    <button
      onClick={onToggle}
      className={clsx(
        'flex w-full items-center justify-between px-4 py-3',
        'text-left text-xs font-semibold uppercase tracking-[0.3em] text-slate-400',
        'hover:bg-slate-50 transition-colors',
        className
      )}
    >
      <span className="flex items-center gap-2">
        {title}
        {badges}
      </span>
      <ChevronDown className={clsx('h-4 w-4 transition-transform', expanded && 'rotate-180')} />
    </button>
  );
}

interface CloseButtonProps {
  onClick: () => void;
  title?: string;
  className?: string;
  iconClassName?: string;
}

export function CloseButton({
  onClick,
  title = 'Close',
  className,
  iconClassName = 'w-4 h-4',
}: CloseButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'p-1 rounded text-slate-400 hover:text-slate-600 hover:bg-slate-100 transition-colors',
        className
      )}
      title={title}
    >
      <X className={iconClassName} />
    </button>
  );
}

