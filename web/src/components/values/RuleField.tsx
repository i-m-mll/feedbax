import { Autocomplete } from '@base-ui/react/autocomplete';
import clsx from 'clsx';
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { FieldProblem } from '@/components/ui/FieldProblem';
import { FieldRow } from '@/components/ui/FieldRow';
import { TermText } from '@/components/ui/TextField';
import { ContextMenu, ContextMenuContent, ContextMenuTrigger, Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Button, IconButton } from '@/components/ui/Button';
import { X } from 'lucide-react';
import { usePopupLayer } from '@/components/ui/popupLayer';
import { fieldBox, fieldInput, popupItem, popupSurface, primaryText } from '@/components/ui/styles';
import { contextMenuKey } from '@/components/pipeline/CommandContextMenu';
import { rulePickSurface } from '@/features/projectWindow/elementSelection';
import { useDraft } from '@/hooks/useDraft';
import { ruleCompletions, type RuleCompletion, type RuleVocabulary } from './runRule';

/** A rule's problem, and the term it is about in the text it is about, drawn while the field shows that text. */
export interface RuleFieldProblem {
  readonly reason: string;
  readonly term?: { readonly text: string; readonly start: number; readonly end: number };
  /** What the mark offers in place of what the rule names that no longer exists. */
  readonly offers?: readonly RuleFieldCommand[];
}

/** One clause of a choice of several: its text, its problem, and the edit that removes it. */
export interface RuleFieldClause {
  readonly text: string;
  readonly problem?: RuleFieldProblem;
  remove?(): void;
}

/** A problem's mark; one that offers replacements opens them as a menu. */
function ProblemMark({ problem, id }: { problem: RuleFieldProblem; id: string }) {
  const mark = <FieldProblem reason={problem.reason} id={id} className="mr-1" />;
  return problem.offers?.length ? <Menu>
    <MenuTrigger render={<Button variant="bare" size="content" aria-label={studioName('problem')} aria-describedby={id} />}>{mark}</MenuTrigger>
    <MenuContent align="end">
      {problem.offers.map((offer) => <MenuItem key={offer.label} disabled={offer.disabled} onClick={offer.invoke}>{offer.label}</MenuItem>)}
    </MenuContent>
  </Menu> : mark;
}

/** A command on the field's menu. */
export interface RuleFieldCommand {
  readonly label: string;
  readonly disabled?: boolean;
  invoke(): void;
}

export interface RuleFieldProps {
  label: string;
  /** The saved rule. */
  value: string;
  vocabulary: RuleVocabulary;
  /** The trailing mark: how many runs the rule chooses, of how many (`4/16`). */
  count?: string;
  /** The pinned axes or measured ranks, by name. */
  pinned?: readonly string[];
  /** A choice of several clauses, one line each beneath the field, with the clause's problem and its removal. */
  clauses?: readonly RuleFieldClause[];
  /** Why the rule being written, or the saved one, does not choose runs, with the term of that text it is about. */
  problem?: RuleFieldProblem;
  disabled?: boolean;
  commands?: readonly RuleFieldCommand[];
  /** While runs are picked for the rule, what the picked runs write: the rule's text, or the rows they pin; a new object for each pick. */
  picked?: { readonly text: string; readonly pinned?: readonly string[]; readonly representable?: boolean };
  /** Ends picking: saving what was picked on Enter or on leaving the field and what it picks from, restoring on Escape. */
  onPickEnd?(save: boolean): void;
  /** The rule as it is being written while the field has focus; null once it is left. */
  onDraft?(rule: string | null): void;
  /** Saves the rule, on Enter or on leaving the field; the saved rule stays shown until the caller's value changes. */
  onCommit(rule: string): void;
}

/**
 * A run rule on one line: completion offers axes, their operators and values, and where the rule ranks, lowest and
 * highest with the result names after them, once the rule is written in or ArrowDown is pressed. Enter or leaving the
 * field saves; Escape restores the saved rule.
 */
export function RuleField({
  label, value, vocabulary, count, pinned, clauses, problem, disabled, commands, picked, onPickEnd, onDraft, onCommit,
}: RuleFieldProps) {
  const id = useId();
  const [draft, setDraft] = useDraft(value);
  const [caret, setCaret] = useState(value.length);
  const [focused, setFocused] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const input = useRef<HTMLInputElement | null>(null);
  const [element, setElement] = useState<HTMLInputElement | null>(null);
  const highlighted = useRef<RuleCompletion>();
  const placed = useRef<number>();
  const box = useRef<HTMLDivElement>(null);
  /** The saved rule when picked runs were last saved, until which leaving the field saves nothing more. */
  const pickSaved = useRef<string>();
  const picking = picked !== undefined;
  const canSavePick = picked?.representable !== false;
  const completions = useMemo(() => focused && !picking ? ruleCompletions(draft, caret, vocabulary) : [],
    [focused, picking, draft, caret, vocabulary]);
  const endPick = useRef(onPickEnd);
  endPick.current = onPickEnd;
  useEffect(() => {
    if (!picking) return;
    const end = (save: boolean) => {
      if (save && canSavePick) pickSaved.current = value;
      else setDraft(value);
      endPick.current?.(save && canSavePick);
    };
    // An axis's values are picked from its menu, which is drawn outside the card.
    const leaves = (target: EventTarget | null) => target instanceof Element
      && !box.current?.contains(target) && !target.closest(`[${rulePickSurface}], [role="menu"]`);
    const press = (event: Event) => { if (leaves(event.target)) end(true); };
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !(event.target instanceof Element && event.target.closest('[role="menu"]'))) end(false);
      else if (event.key === 'Enter' && event.target instanceof Node && box.current?.contains(event.target)) {
        event.preventDefault();
        if (canSavePick) end(true);
      }
    };
    document.addEventListener('pointerdown', press, true);
    document.addEventListener('focusin', press, true);
    document.addEventListener('keydown', key, true);
    return () => {
      document.removeEventListener('pointerdown', press, true);
      document.removeEventListener('focusin', press, true);
      document.removeEventListener('keydown', key, true);
    };
  }, [picking, value, canSavePick]);
  // Each pick, including one from an axis's menu, hands the keyboard back to the field, so Enter saves what is picked.
  useEffect(() => {
    if (!picked) return;
    setDraft(picked.text);
    input.current?.focus();
  }, [picked]);
  useLayoutEffect(() => {
    if (placed.current === undefined || !input.current) return;
    input.current.setSelectionRange(placed.current, placed.current);
    placed.current = undefined;
  });
  function write(text: string, at: number) {
    pickSaved.current = undefined;
    setDraft(text);
    setCaret(at);
    setDismissed(false);
    onDraft?.(text);
  }
  function commit() {
    if (picking || pickSaved.current === value) return;
    pickSaved.current = undefined;
    if (!disabled && draft.trim() !== value.trim()) onCommit(draft.trim());
  }
  const shownPinned = picked ? picked.pinned : pinned;
  const term = problem?.term && problem.term.text === draft && problem.term.end <= draft.length
    ? { ...problem.term, reason: problem.reason } : undefined;
  const field = <div ref={box} className={clsx(fieldBox, 'min-w-0', term && 'relative')}>
    <Autocomplete.Input ref={(node: HTMLInputElement | null) => { input.current = node; setElement(node); }} id={id} disabled={disabled} spellCheck={false} autoComplete="off"
      aria-invalid={problem !== undefined || undefined} aria-describedby={problem ? `${id}-problem` : undefined}
      className={clsx(fieldInput, 'font-mono', term && 'text-transparent caret-shell-foreground')}
      onSelect={(event) => setCaret(event.currentTarget.selectionStart ?? draft.length)}
      onFocus={() => { setFocused(true); setDismissed(true); onDraft?.(draft); }}
      onBlur={() => { setFocused(false); onDraft?.(null); commit(); }}
      onKeyDown={(event) => {
        if (contextMenuKey(event) || event.nativeEvent.isComposing || picking) return;
        if (event.key === 'Enter' && !highlighted.current) {
          event.preventDefault();
          setDismissed(true);
          commit();
        } else if (event.key === 'Escape' && (dismissed || !completions.length) && draft !== value) {
          // The list's own Escape, with the list closed, empties the field; here it restores the saved rule.
          event.stopPropagation();
        }
      }} />
    {shownPinned?.length
      ? <span className="shrink-0 pr-1 text-shell-muted" aria-description={shownPinned.join(', ')}>{studioName('rows')}</span> : null}
    {count ? <span className="shrink-0 pr-control-inline tabular-nums text-shell-muted">{count}</span> : null}
    {term && element && <TermText input={element} text={draft} term={term} />}
    {problem && <ProblemMark problem={problem} id={`${id}-problem`} />}
  </div>;
  const lines = !picking && clauses && clauses.length > 1 && <ul aria-label={studioName('rows')} className="grid gap-0.5">
    {clauses.map((clause, index) => <li key={clause.text} className="flex min-w-0 items-center gap-1 font-mono text-xs">
      <span className="min-w-0 truncate">{clause.text}</span>
      {clause.problem && <ProblemMark problem={clause.problem} id={`${id}-clause-${index}`} />}
      {clause.remove && <IconButton label={studioAccessibleName('remove', clause.text)} icon={<X aria-hidden="true" />}
        disabled={disabled} onClick={clause.remove} />}
    </li>)}
  </ul>;
  return <FieldRow label={label} htmlFor={id} below={lines || undefined}>
    <Autocomplete.Root items={completions} filter={null} value={draft} mode="list"
      itemToStringValue={(item: RuleCompletion) => item.text}
      open={focused && !dismissed && completions.length > 0}
      onOpenChange={(open, details) => { if (open) setDismissed(false); else if (details.reason !== 'item-press') setDismissed(true); }}
      onItemHighlighted={(item: RuleCompletion | undefined) => { highlighted.current = item; }}
      onValueChange={(text, details) => {
        if (picking) {
          if (details.reason === 'escape-key') return;
          onPickEnd?.(false);
        }
        if (details.reason === 'escape-key') {
          placed.current = value.length;
          write(value, value.length);
          return setDismissed(true);
        }
        const chosen = details.reason === 'item-press' ? completions.find((item) => item.text === text) : undefined;
        if (chosen) placed.current = chosen.caret;
        write(text, chosen?.caret ?? input.current?.selectionStart ?? text.length);
      }}>
      {commands?.length ? <ContextMenu>
        <ContextMenuTrigger render={field} />
        <ContextMenuContent>
          {commands.map((command) => <MenuItem key={command.label} disabled={command.disabled} onClick={command.invoke}>
            {command.label}
          </MenuItem>)}
        </ContextMenuContent>
      </ContextMenu> : field}
      <Autocomplete.Portal container={usePopupLayer()}>
        <Autocomplete.Positioner className="z-50 outline-none" sideOffset={4} align="start">
          <Autocomplete.Popup className={clsx(popupSurface, 'w-max min-w-[var(--anchor-width)] max-w-[var(--available-width)]')}>
            <Autocomplete.List className="max-h-[min(22rem,var(--available-height))] overflow-y-auto">
              {(item: RuleCompletion) => <Autocomplete.Item key={item.text} value={item} className={popupItem}>
                <span className={clsx(primaryText, 'font-mono')}>{item.label}</span>
              </Autocomplete.Item>}
            </Autocomplete.List>
          </Autocomplete.Popup>
        </Autocomplete.Positioner>
      </Autocomplete.Portal>
    </Autocomplete.Root>
  </FieldRow>;
}

