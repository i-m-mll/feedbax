import { Input } from '@base-ui/react/input';
import clsx from 'clsx';
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';

export interface InlineEditProps {
  value: string;
  editing: boolean;
  onEditingChange: (editing: boolean) => void;
  /** Return false to refuse the text and stay in editing; on blur a refused text is dropped. */
  onCommit: (value: string) => boolean | void;
  children?: ReactNode;
  'aria-label': string;
  className?: string;
}

export function InlineEdit({ value, editing, onEditingChange, onCommit, children, className, ...aria }: InlineEditProps) {
  const [draft, setDraft] = useState(value);
  const input = useRef<HTMLInputElement>(null);
  const settled = useRef(false);

  useLayoutEffect(() => {
    if (!editing) return;
    settled.current = false;
    setDraft(value);
    input.current?.focus();
    input.current?.select();
  }, [editing, value]);

  if (!editing) return <span className={clsx('min-w-0 truncate', className)}>{children ?? value}</span>;

  const commit = () => {
    const next = draft.trim();
    if (next === value) return finish();
    if (!next || onCommit(next) === false) return input.current?.select();
    finish();
  };
  const finish = () => {
    settled.current = true;
    onEditingChange(false);
  };

  return (
    <Input
      ref={input}
      value={draft}
      onValueChange={setDraft}
      aria-invalid={!draft.trim() || undefined}
      className={clsx(
        'h-[calc(var(--row-height)-4px)] w-full min-w-0 rounded-row border border-shell-control bg-shell-surface px-1',
        'text-[length:inherit] font-[inherit] text-shell-foreground outline-none',
        'focus:outline focus:outline-2 focus:-outline-offset-1 focus:outline-shell-accent',
        'aria-[invalid]:border-shell-status-failure',
        className,
      )}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === 'Enter') {
          event.preventDefault();
          commit();
        } else if (event.key === 'Escape') {
          event.preventDefault();
          finish();
        }
      }}
      onBlur={() => {
        if (settled.current) return;
        const next = draft.trim();
        if (next && next !== value) onCommit(next);
        finish();
      }}
      onClick={(event) => event.stopPropagation()}
      onDoubleClick={(event) => event.stopPropagation()}
      {...aria}
    />
  );
}
