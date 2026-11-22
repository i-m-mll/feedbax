import { TriangleAlert, X } from 'lucide-react';
import { useId, useLayoutEffect, useRef, type ReactNode } from 'react';
import { appTip } from '@window-shell/browser/tooltip';
import { studioName, studioPlaceName, studioProblemDetail, studioRunReason, type StudioPlace } from '@/components/activity/studioVocabulary';
import type { StudioProjectWindowMapProblem } from '@/generated/studioContracts';
import { commandAttributes, type StudioCommand } from './studioCommands';
import { Button, IconButton } from '@/components/ui/Button';

/** A reason that names a place: pressed, it reveals the place. Every list of problems and changes draws this row. */
export function MentionRow({ place, detail, context = [], placeShown = true, refusal, onReveal }: {
  place: StudioPlace | null;
  detail: string;
  /** Why the last gesture on this entry was refused. */
  refusal?: string;
  /** False where the surrounding heading or region already names the place. */
  placeShown?: boolean;
  /** What the surrounding list does not already say, such as the step or variant. */
  context?: readonly string[];
  onReveal?: () => void;
}) {
  const entry = <>
    {placeShown && <><span className="font-medium">{studioPlaceName(place)}</span>{' · '}</>}{detail}
    {context.length > 0 && <span className="block text-xs text-shell-muted">{context.join(' · ')}</span>}
  </>;
  const refused = refusal ? { 'aria-description': refusal, ...appTip(studioRunReason({ reason: refusal })) } : {};
  return onReveal
    ? <Button variant="ghost" size="content" className="block w-full rounded-control px-control-inline py-1 text-left" onClick={onReveal} {...refused}>
      {entry}
    </Button>
    : <div className="px-control-inline py-1" {...refused}>{entry}</div>;
}

export function ProblemRow({ problem, ...row }: {
  problem: StudioProjectWindowMapProblem;
  context?: readonly string[];
  placeShown?: boolean;
  refusal?: string;
  onReveal?: () => void;
}) {
  return <MentionRow place={problem.place} detail={studioProblemDetail(problem)} {...row} />;
}

/** A problem as the Problems list holds it: one per place and detail, with the number of runs it is on. */
export interface ProblemEntry {
  readonly problem: StudioProjectWindowMapProblem;
  readonly runs: number;
}

/**
 * The entries of the Problems list. A problem repeated across a step's rows is one entry; a run waiting on another
 * step's run is not listed, its own row and step mark saying so.
 */
export function problemEntries(problems: readonly StudioProjectWindowMapProblem[]): ProblemEntry[] {
  const entries = new Map<string, { problem: StudioProjectWindowMapProblem; rows: Set<string | null> }>();
  for (const problem of problems) {
    if (problem.code === 'blocked_input') continue;
    const key = JSON.stringify([problem.step, problem.variant, problem.place, problem.path, problem.detail]);
    const entry = entries.get(key) ?? { problem, rows: new Set() };
    entry.rows.add(problem.row);
    entries.set(key, entry);
  }
  return [...entries.values()].map(({ problem, rows }) => ({ problem, runs: rows.size }));
}

export function StudioProblems({ command, count, open, opener, onOpen, onClose, children }: {
  command: StudioCommand;
  count: number;
  open: boolean;
  opener: HTMLElement | null;
  onOpen: (opener: HTMLElement) => void;
  onClose: () => void;
  children: ReactNode;
}) {
  const id = useId();
  const anchor = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const dismiss = useRef(onClose);
  dismiss.current = onClose;

  function closeAndRestore() {
    dismiss.current();
    if (opener?.isConnected && opener.getClientRects().length && !opener.matches(':disabled') && !opener.closest('[inert], [hidden]')) {
      opener.focus();
    } else {
      trigger.current?.focus();
    }
  }

  useLayoutEffect(() => {
    const entry = anchor.current;
    const surface = panel.current;
    if (!open || !entry || !surface) return;
    const fit = () => {
      const rect = entry.getBoundingClientRect();
      surface.style.maxInlineSize = `${Math.max(0, rect.right - 8)}px`;
      surface.style.maxBlockSize = `${Math.max(0, rect.top - 16)}px`;
    };
    fit();
    const resize = new ResizeObserver(fit);
    resize.observe(entry);
    if (entry.parentElement) resize.observe(entry.parentElement);
    window.addEventListener('resize', fit);
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !entry.contains(event.target)) dismiss.current();
    };
    document.addEventListener('pointerdown', outside, true);
    close.current?.focus();
    return () => {
      resize.disconnect();
      window.removeEventListener('resize', fit);
      document.removeEventListener('pointerdown', outside, true);
    };
  }, [open, opener]);

  return (
    <div ref={anchor} className="relative"
      onBlur={(event) => {
        if (open && event.relatedTarget instanceof Node && !event.currentTarget.contains(event.relatedTarget)) dismiss.current();
      }}
      onKeyDown={(event) => {
        if (!open || event.key !== 'Escape') return;
        event.preventDefault();
        event.stopPropagation();
        closeAndRestore();
      }}>
      <Button ref={trigger} variant="ghost" {...commandAttributes(command)} aria-label={undefined} aria-haspopup="dialog"
        aria-expanded={open} aria-controls={open ? id : undefined}
        onClick={(event) => onOpen(event.currentTarget)} className="font-medium">
        <TriangleAlert aria-hidden="true" />
        {studioName('problems')} <span className="tabular-nums">{count}</span>
      </Button>
      {open && (
        <section ref={panel} id={id} role="dialog" aria-labelledby={`${id}-title`}
          className="absolute bottom-full right-0 z-50 mb-2 grid w-[34rem] gap-3 overflow-auto rounded-shell-control border border-shell-divider bg-shell-chrome p-4 text-shell-foreground shadow-xl">
          <header className="flex items-center justify-between gap-2">
            <h2 id={`${id}-title`} className="font-semibold">{studioName('problems')}</h2>
            <IconButton ref={close} label={studioName('close')} icon={<X aria-hidden="true" />} onClick={closeAndRestore} />
          </header>
          {children}
        </section>
      )}
    </div>
  );
}
