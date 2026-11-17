import clsx from 'clsx';
import type { ReactNode } from 'react';

export interface PanelHeaderProps {
  title: ReactNode;
  glyph?: ReactNode;
  actions?: ReactNode;
  className?: string;
}

export function PanelHeader({ title, glyph, actions, className }: PanelHeaderProps) {
  return (
    <header
      className={clsx(
        'flex h-panel-header shrink-0 items-center gap-2 border-b border-shell-divider px-panel-header-inline text-shell-foreground [&_svg]:size-[1.15em]',
        className,
      )}
    >
      {glyph ? <span className="flex text-shell-muted">{glyph}</span> : null}
      <h2 className="min-w-0 flex-1 truncate text-base font-medium">{title}</h2>
      {actions ? <div className="flex shrink-0 items-center gap-0.5">{actions}</div> : null}
    </header>
  );
}
