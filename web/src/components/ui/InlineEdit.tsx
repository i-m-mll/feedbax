import { Input } from '@base-ui/react/input';
import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react';

export interface InlineEditProps {
  value: string;
  editing: boolean;
  onEditingChange: (editing: boolean) => void;
  /** Return false to refuse the text and stay in editing; on blur a refused text is dropped. */
  onCommit: (value: string) => boolean | void;
  /** Why a changed text cannot be committed: it is marked as typed, with the reason as its tip, and Enter keeps it. */
  unavailable?: (value: string) => string | undefined;
  children?: ReactNode;
  'aria-label': string;
  className?: string;
  /**
   * In a table cell the name is always its input, filling the cell, so a click into it edits it; editing follows
   * the input's focus, and `editing` set from outside focuses it with the name selected.
   */
  form?: 'cell';
  /** Shown while the text is empty, which then commits nothing rather than being refused. */
  empty?: string;
}

export function InlineEdit({
  value, editing, onEditingChange, onCommit, unavailable, children, className, form, empty, ...aria
}: InlineEditProps) {
  const [draft, setDraft] = useState(value);
  const input = useRef<HTMLInputElement>(null);
  const settled = useRef(false);

  const cell = form === 'cell';
  useLayoutEffect(() => {
    if (!editing) {
      if (cell) setDraft(value);
      return;
    }
    settled.current = false;
    if (cell && document.activeElement === input.current) return;
    setDraft(value);
    input.current?.focus();
    input.current?.select();
  }, [editing, value]);

  if (!editing && !cell) return <span className={clsx('min-w-0 truncate', className)}>{children ?? value}</span>;

  const typed = draft.trim();
  const reason = typed && typed !== value ? unavailable?.(typed) : undefined;
  const commit = () => {
    const next = draft.trim();
    if (next === value) return finish(true);
    if (!next && empty !== undefined) return finish(true);
    if (!next || reason || onCommit(next) === false) return input.current?.select();
    finish();
  };
  const finish = (revert = false) => {
    settled.current = true;
    if (cell) {
      if (revert) setDraft(value);
      input.current?.blur();
    }
    onEditingChange(false);
  };

  return (
    <Input
      ref={input}
      value={draft}
      onValueChange={setDraft}
      placeholder={empty}
      aria-invalid={(empty === undefined && !typed) || !!reason || undefined}
      {...appTip(reason)}
      className={clsx(
        'w-full min-w-0 text-[length:inherit] font-[inherit] text-shell-foreground outline-none',
        cell ? 'studio-cell-input block h-[calc(var(--row-height)-1px)] truncate text-left bg-transparent px-row-inline aria-[invalid]:text-shell-status-failure'
          : clsx('h-[calc(var(--row-height)-4px)] rounded-row border border-shell-control bg-shell-surface px-1',
            'focus:outline focus:outline-2 focus:-outline-offset-1 focus:outline-shell-accent aria-[invalid]:border-shell-status-failure'),
        className,
      )}
      onFocus={cell ? () => {
        settled.current = false;
        if (!editing) onEditingChange(true);
      } : undefined}
      onKeyDown={(event) => {
        event.stopPropagation();
        if (event.key === 'Enter') {
          event.preventDefault();
          commit();
        } else if (event.key === 'Escape') {
          event.preventDefault();
          finish(true);
        }
      }}
      onBlur={() => {
        if (settled.current) return;
        const next = draft.trim();
        finish(!(next && next !== value && !reason && onCommit(next) !== false));
      }}
      onClick={(event) => event.stopPropagation()}
      onDoubleClick={(event) => event.stopPropagation()}
      {...aria}
    />
  );
}
