import { Combobox } from '@base-ui/react/combobox';
import clsx from 'clsx';
import { Check, ChevronDown, Crosshair, Plus } from 'lucide-react';
import { useEffect, useMemo, useRef, type ReactNode } from 'react';
import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { usePopupLayer } from '@/components/ui/popupLayer';
import { disabledLook, fieldFrame, fieldInput, focusRing, glyph, leadingSlot, popupItem, popupSurface } from '@/components/ui/styles';
import {
  pickKey, usePick, type ElementSelectionStore, type PickTarget, type ReferencePick,
} from '@/features/projectWindow/elementSelection';

/** A thing a reference can read: a document, a step's output, or an element of a document. */
export interface ReferenceChoice {
  readonly id: string;
  readonly label: string;
  readonly glyph?: ReactNode;
  /** The card or element a click fills the field with while picking. */
  readonly pick?: PickTarget;
}

export interface ReferenceGroup {
  readonly label: string;
  readonly choices: readonly ReferenceChoice[];
}

export interface ReferenceFieldProps {
  /** The field's name, for its controls' accessible names. */
  label: string;
  groups: readonly ReferenceGroup[];
  value: ReferenceChoice | null;
  onChoose?: (choice: ReferenceChoice) => void;
  /** The window's element selection, whose pick the crosshair arms. Without it the field has no crosshair. */
  picking?: ElementSelectionStore;
  /** The visible label that names the field's trigger, popup and filter. */
  'aria-labelledby': string;
  /** The field adds a selection to a list: it holds no value, and its trigger is the list's `+`. */
  adds?: boolean;
  className?: string;
}

/**
 * The field for which thing a step or element reads. Closed, the chosen thing's glyph and name; open, the things
 * the field accepts, grouped, filtered by typing. Its crosshair arms picking: a click on an accepted card or
 * element fills it, and Escape disarms it.
 */
export function ReferenceField({ label, groups, value, onChoose, picking, adds, className, ...aria }: ReferenceFieldProps) {
  const shown = useMemo(() => groups.filter(({ choices }) => choices.length), [groups]);
  const pickable = useMemo(() => new Map(groups.flatMap(({ choices }) =>
    choices.flatMap((choice) => choice.pick ? [[pickKey(choice.pick), choice] as const] : []))), [groups]);
  const armed = usePickArming(picking, onChoose && pickable.size ? pickable : undefined, onChoose);
  const disabled = !onChoose;
  return <div className={clsx('flex min-w-0 items-center gap-1', className)}>
    <Combobox.Root<ReferenceChoice>
      items={shown.map(({ label, choices }) => ({ value: label, items: choices }))}
      value={value}
      onValueChange={(next: ReferenceChoice | null) => { if (next && next.id !== value?.id) onChoose?.(next); }}
      isItemEqualToValue={(item: ReferenceChoice, current: ReferenceChoice) => item.id === current.id}
      itemToStringLabel={(item: ReferenceChoice) => item.label}
      itemToStringValue={(item: ReferenceChoice) => item.id}
      disabled={disabled}
      autoHighlight
    >
      <Combobox.Trigger {...(adds ? { 'aria-label': studioAccessibleName('add', label) } : aria)}
        className={clsx(fieldFrame, 'h-control min-w-0 flex-1 cursor-default select-none items-center gap-1 pr-control-inline',
          !value?.glyph && 'pl-control-inline',
          'hover:[&:not([data-disabled])]:bg-shell-hover data-[popup-open]:bg-shell-hover', focusRing, glyph, disabledLook)}>
        {value?.glyph && <span aria-hidden="true" className={leadingSlot}>{value.glyph}</span>}
        {adds ? <Plus aria-hidden="true" className="text-shell-muted" />
          : <span className="min-w-0 flex-1 truncate text-left">{value?.label ?? '—'}</span>}
        {!adds && <ChevronDown aria-hidden="true" className="text-shell-muted" />}
      </Combobox.Trigger>
      <Combobox.Portal container={usePopupLayer()}>
        <Combobox.Positioner className="z-50 outline-none" sideOffset={4} align="start">
          <Combobox.Popup aria-labelledby={aria['aria-labelledby']}
            className={clsx(popupSurface, 'grid min-w-[var(--anchor-width)] max-w-[var(--available-width)] gap-1')}>
            <Combobox.Input aria-labelledby={aria['aria-labelledby']} className={clsx(fieldFrame, fieldInput, 'h-control')} />
            <Combobox.List className="max-h-[min(22rem,var(--available-height))] overflow-y-auto empty:hidden">
              {(group: { value: string; items: readonly ReferenceChoice[] }) => (
                <Combobox.Group key={group.value} items={group.items}>
                  <Combobox.GroupLabel className="px-row-inline pt-1 text-xs text-shell-muted">{group.value}</Combobox.GroupLabel>
                  <Combobox.Collection>
                    {(choice: ReferenceChoice) => (
                      <Combobox.Item key={choice.id} value={choice} className={clsx(popupItem, 'data-[selected]:font-medium')}>
                        <span className="flex w-[1.15em] justify-center">
                          <Combobox.ItemIndicator className="flex"><Check /></Combobox.ItemIndicator>
                        </span>
                        {choice.glyph && <span aria-hidden="true" className="flex text-shell-muted">{choice.glyph}</span>}
                        <span className="flex-1 truncate">{choice.label}</span>
                      </Combobox.Item>
                    )}
                  </Combobox.Collection>
                </Combobox.Group>
              )}
            </Combobox.List>
          </Combobox.Popup>
        </Combobox.Positioner>
      </Combobox.Portal>
    </Combobox.Root>
    {picking && pickable.size > 0 && <IconButton label={studioAccessibleName('pick', adds ? studioAccessibleName('add', label) : label)} icon={<Crosshair />}
      pressed={armed.on} disabled={disabled} onClick={armed.toggle} />}
  </div>;
}

/** Arms and disarms this field's pick in the window's selection; the armed pick is the field's own until it is used. */
function usePickArming(
  store: ElementSelectionStore | undefined,
  pickable: ReadonlyMap<string, ReferenceChoice> | undefined,
  onChoose: ((choice: ReferenceChoice) => void) | undefined,
) {
  const own = useRef<ReferencePick | null>(null);
  const current = useRef({ pickable, onChoose });
  current.current = { pickable, onChoose };
  const armed = usePick(store);
  const on = !!armed && armed === own.current;
  useEffect(() => {
    if (!on || !store) return;
    const disarm = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      store.setState({ pick: null });
    };
    window.addEventListener('keydown', disarm, true);
    return () => window.removeEventListener('keydown', disarm, true);
  }, [on, store]);
  useEffect(() => () => {
    if (store && own.current && store.getState().pick === own.current) store.setState({ pick: null });
  }, [store]);
  useEffect(() => {
    if (on && !pickable) store?.setState({ pick: null });
  }, [on, pickable, store]);
  return {
    on,
    toggle: () => {
      if (!store) return;
      if (on) {
        store.setState({ pick: null });
        return;
      }
      const pick: ReferencePick = {
        accepts: (target) => !!current.current.pickable?.has(pickKey(target)),
        choose: (target) => {
          const choice = current.current.pickable?.get(pickKey(target));
          if (choice) current.current.onChoose?.(choice);
        },
      };
      own.current = pick;
      store.setState({ pick });
    },
  };
}
