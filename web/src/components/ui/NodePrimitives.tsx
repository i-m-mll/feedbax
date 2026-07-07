import { Handle, type HandleProps } from '@xyflow/react';
import clsx from 'clsx';
import type { CSSProperties, MouseEventHandler, ReactNode } from 'react';
import { semanticTokens } from './semanticTokens';

export type NodeTone =
  | 'model'
  | 'subgraph'
  | 'analysis'
  | 'dependency'
  | 'dataSource'
  | 'transform';

const nodeToneClasses: Record<NodeTone, { shell: string; border: string; header: string }> = {
  model: {
    shell: 'rounded-xl border-2 shadow-soft bg-white/90 backdrop-blur',
    border: 'border-slate-200',
    header: 'bg-slate-50/70 border-slate-100 rounded-t-xl',
  },
  subgraph: {
    shell: 'rounded-xl border shadow-soft bg-white/90 backdrop-blur',
    border: 'border-violet-200',
    header: 'bg-violet-50/80 border-violet-100 rounded-t-xl',
  },
  analysis: {
    shell: 'rounded-xl border shadow-soft bg-white/90 backdrop-blur',
    border: 'border-emerald-200',
    header: 'bg-emerald-50/60 border-emerald-100/60 rounded-t-xl',
  },
  dependency: {
    shell: 'rounded-xl border shadow-soft bg-white/70 backdrop-blur',
    border: 'border-slate-200/60',
    header: 'bg-slate-50/60 border-slate-100/60 rounded-t-xl',
  },
  dataSource: {
    shell: 'rounded-lg border bg-slate-50/80 backdrop-blur shadow-soft',
    border: 'border-slate-200/80',
    header: 'border-slate-100/80 rounded-t-lg',
  },
  transform: {
    shell: 'rounded-full border bg-white/80 backdrop-blur shadow-soft',
    border: 'border-slate-200/80',
    header: '',
  },
};

interface NodeShellProps {
  tone?: NodeTone;
  selected?: boolean;
  selectedRing?: 'normal' | 'strong';
  highlighted?: boolean;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}

export function NodeShell({
  tone = 'model',
  selected = false,
  selectedRing = 'normal',
  highlighted = false,
  className,
  style,
  children,
}: NodeShellProps) {
  const classes = nodeToneClasses[tone];
  return (
    <div
      className={clsx(
        'relative transition-all duration-150',
        classes.shell,
        selected
          ? clsx(
              semanticTokens.selected.border,
              selectedRing === 'strong' ? 'ring-2' : 'ring-1',
              selectedRing === 'strong'
                ? semanticTokens.selected.strongRing
                : semanticTokens.selected.ring
            )
          : classes.border,
        highlighted &&
          !selected &&
          clsx(semanticTokens.highlighted.border, 'ring-2', semanticTokens.highlighted.ring),
        className
      )}
      style={style}
    >
      {children}
    </div>
  );
}

interface NodeHeaderProps {
  tone?: NodeTone;
  collapsed?: boolean;
  className?: string;
  style?: CSSProperties;
  onDoubleClick?: MouseEventHandler<HTMLDivElement>;
  children: ReactNode;
}

export function NodeHeader({
  tone = 'model',
  collapsed = false,
  className,
  style,
  onDoubleClick,
  children,
}: NodeHeaderProps) {
  const classes = nodeToneClasses[tone];
  return (
    <div
      className={clsx(
        'px-3 py-2 flex items-center gap-3 overflow-hidden',
        !collapsed && 'border-b',
        collapsed && tone === 'model' ? 'rounded-xl' : classes.header,
        className
      )}
      style={style}
      onDoubleClick={onDoubleClick}
    >
      {children}
    </div>
  );
}

export type PortTone =
  | 'model'
  | 'state'
  | 'analysis'
  | 'dependency'
  | 'dynamic'
  | 'task'
  | 'objective'
  | 'selected'
  | 'highlighted';

type PortSize = 'xs' | 'sm' | 'md' | 'lg';

const portSizeClasses: Record<PortSize, string> = {
  xs: 'w-1.5 h-1.5',
  sm: 'w-2 h-2',
  md: 'w-2.5 h-2.5',
  lg: 'w-3 h-3',
};

const portToneClasses: Record<PortTone, string> = {
  model: 'bg-slate-400',
  state: 'bg-slate-600',
  analysis: 'bg-emerald-400',
  dependency: 'bg-slate-300',
  dynamic: 'bg-white ring-1 ring-slate-300 border-slate-300',
  task: clsx(semanticTokens.task.fill, 'ring-2', semanticTokens.task.ring),
  objective: clsx(semanticTokens.objective.fill, 'ring-2', semanticTokens.objective.ring),
  selected: clsx(semanticTokens.selected.fill, 'ring-4', semanticTokens.selected.softRing),
  highlighted: clsx(semanticTokens.highlighted.fill, 'ring-2', semanticTokens.highlighted.ring),
};

interface PortHandleProps extends HandleProps {
  tone?: PortTone;
  size?: PortSize;
  arrow?: 'left' | 'right';
}

export function PortHandle({
  tone = 'model',
  size = 'sm',
  arrow,
  className,
  style,
  ...props
}: PortHandleProps) {
  const arrowClip =
    arrow === 'right'
      ? 'polygon(100% 0%, 0% 50%, 100% 100%)'
      : arrow === 'left'
        ? 'polygon(0% 0%, 100% 50%, 0% 100%)'
        : undefined;
  return (
    <Handle
      {...props}
      style={{ ...style, clipPath: style?.clipPath ?? arrowClip }}
      className={clsx(
        portSizeClasses[size],
        'z-20 border border-white shadow-soft transition-all duration-150',
        portToneClasses[tone],
        className
      )}
    />
  );
}
