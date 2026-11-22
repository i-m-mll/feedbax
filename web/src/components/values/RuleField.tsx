import { Autocomplete } from '@base-ui/react/autocomplete';
import clsx from 'clsx';
import { useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { FieldRow } from '@/components/ui/FieldRow';
import { ContextMenu, ContextMenuContent, ContextMenuTrigger, MenuItem } from '@/components/ui/Menu';
import { usePopupLayer } from '@/components/ui/popupLayer';
import { fieldBox, fieldInput, popupItem, popupSurface, primaryText } from '@/components/ui/styles';
import { contextMenuKey } from '@/components/pipeline/CommandContextMenu';
import { useDraft } from '@/hooks/useDraft';
import { ruleCompletions, type RuleCompletion, type RuleVocabulary } from './runRule';

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
  /** The rows To Rows pinned beside the rule, by name. */
  pinned?: readonly string[];
  /** Why the rule, or the last one committed, is refused. */
  refusal?: string;
  disabled?: boolean;
  commands?: readonly RuleFieldCommand[];
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
export function RuleField({ label, value, vocabulary, count, pinned, refusal, disabled, commands, onDraft, onCommit }: RuleFieldProps) {
  const id = useId();
  const [draft, setDraft] = useDraft(value);
  const [caret, setCaret] = useState(value.length);
  const [focused, setFocused] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const highlighted = useRef<RuleCompletion>();
  const placed = useRef<number>();
  const completions = useMemo(() => focused ? ruleCompletions(draft, caret, vocabulary) : [], [focused, draft, caret, vocabulary]);
  useLayoutEffect(() => {
    if (placed.current === undefined || !input.current) return;
    input.current.setSelectionRange(placed.current, placed.current);
    placed.current = undefined;
  });
  function write(text: string, at: number) {
    setDraft(text);
    setCaret(at);
    setDismissed(false);
    onDraft?.(text);
  }
  function commit() {
    if (!disabled && draft.trim() !== value.trim()) onCommit(draft.trim());
  }
  const field = <div className={clsx(fieldBox, 'min-w-0')}>
    <Autocomplete.Input ref={input} id={id} disabled={disabled} spellCheck={false} autoComplete="off"
      aria-invalid={refusal !== undefined || undefined} aria-describedby={refusal ? `${id}-refusal` : undefined}
      className={clsx(fieldInput, 'font-mono')}
      onSelect={(event) => setCaret(event.currentTarget.selectionStart ?? draft.length)}
      onFocus={() => { setFocused(true); setDismissed(true); onDraft?.(draft); }}
      onBlur={() => { setFocused(false); onDraft?.(null); commit(); }}
      onKeyDown={(event) => {
        if (contextMenuKey(event) || event.nativeEvent.isComposing) return;
        if (event.key === 'Enter' && !highlighted.current) {
          event.preventDefault();
          setDismissed(true);
          commit();
        } else if (event.key === 'Escape' && (dismissed || !completions.length) && draft !== value) {
          // The list's own Escape, with the list closed, empties the field; here it restores the saved rule.
          event.stopPropagation();
        }
      }} />
    {pinned?.length ? <span className="shrink-0 pr-1 text-shell-muted" aria-description={pinned.join(', ')}>{studioName('rows')}</span> : null}
    {count ? <span className="shrink-0 pr-control-inline tabular-nums text-shell-muted">{count}</span> : null}
  </div>;
  return <FieldRow label={label} htmlFor={id} below={refusal && <p id={`${id}-refusal`} role="alert">{refusal}</p>}>
    <Autocomplete.Root items={completions} filter={null} value={draft} mode="list"
      itemToStringValue={(item: RuleCompletion) => item.text}
      open={focused && !dismissed && completions.length > 0}
      onOpenChange={(open, details) => { if (open) setDismissed(false); else if (details.reason !== 'item-press') setDismissed(true); }}
      onItemHighlighted={(item: RuleCompletion | undefined) => { highlighted.current = item; }}
      onValueChange={(text, details) => {
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

