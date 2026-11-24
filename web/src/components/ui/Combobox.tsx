import { Combobox as BaseCombobox } from '@base-ui/react/combobox';
import clsx from 'clsx';
import { Check, ChevronDown } from 'lucide-react';
import { useMemo, type ReactNode, type Ref } from 'react';
import type { Option } from './Select';
import { fieldBox, fieldInput, glyph, leadingSlot, popupItem, popupSurface, primaryText, secondaryText } from './styles';
import { usePopupLayer } from './popupLayer';

/**
 * One item chosen; or a set of items checked, which choosing an item checks or unchecks, with the box showing `text`,
 * or with `tokens` each checked item's label as a token.
 */
type ComboboxSelection<Value extends string> =
  | { value: NoInfer<Value> | null; onValueChange: (value: Value | null) => void; checked?: undefined }
  | { checked: readonly NoInfer<Value>[]; onToggle: (value: Value) => void; text: string; tokens?: boolean };

export type ComboboxProps<Value extends string> = ComboboxSelection<Value> & {
  items: readonly Option<Value>[];
  placeholder?: string;
  disabled?: boolean;
  invalid?: boolean;
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-describedby'?: string;
  id?: string;
  /** Drawn without a frame and as wide as the chosen item, inside another field's box. */
  inline?: boolean;
  className?: string;
  /** A glyph at the box's start: the field's kind mark. */
  leading?: ReactNode;
  /** A control at the box's trailing end, before the list's glyph: the changed mark. */
  trailing?: ReactNode;
  inputRef?: Ref<HTMLInputElement>;
};

export function Combobox<Value extends string>(props: ComboboxProps<Value>) {
  const { items, placeholder, disabled, invalid, id, inline, className, leading, trailing, inputRef } = props;
  const aria = { 'aria-label': props['aria-label'], 'aria-labelledby': props['aria-labelledby'] };
  const value = props.checked ? null : props.value;
  const selected = useMemo(() => items.find((item) => item.value === value) ?? null, [items, value]);
  const shown = props.checked ? props.tokens ? '' : props.text : placeholder;
  const indicator = (item: Option<Value>) => props.checked
    ? props.checked.includes(item.value) && <Check />
    : <BaseCombobox.ItemIndicator className="flex"><Check /></BaseCombobox.ItemIndicator>;
  return (
    <BaseCombobox.Root
      items={items}
      value={selected}
      onValueChange={(next: Option<Value> | null) => {
        if (!props.checked) props.onValueChange(next?.value ?? null);
        else if (next) props.onToggle(next.value);
      }}
      isItemEqualToValue={(item: Option<Value>, current: Option<Value>) => item.value === current.value}
      disabled={disabled}
      autoHighlight
    >
      <BaseCombobox.InputGroup data-field-box={inline ? undefined : ''} className={clsx(inline ? 'flex h-full min-w-0 items-center' : fieldBox, 'overflow-hidden', className)}>
        {/* The leading control's own presses, and those in popups it portals, do not open the list. */}
        {leading ? <span className={leadingSlot} onMouseDown={(event) => event.stopPropagation()}>{leading}</span> : null}
        {props.checked && props.tokens && <span className="flex min-w-0 shrink items-center gap-1 overflow-hidden pl-control-inline">
          {items.filter((item) => props.checked!.includes(item.value)).map((item) =>
            <span key={item.value} className="shrink-0 rounded-sm bg-shell-hover px-1 font-mono text-xs leading-5">{item.label}</span>)}
        </span>}
        <BaseCombobox.Input ref={inputRef} id={id} placeholder={shown} aria-invalid={invalid || undefined}
          size={inline ? Math.max(1, (selected?.label ?? shown ?? '').length) : undefined}
          className={clsx(fieldInput, 'text-ellipsis', inline && 'pr-0', leading && 'pl-1',
            props.checked && 'placeholder:text-shell-foreground')} aria-describedby={props['aria-describedby']} {...aria} />
        {trailing ? <span className="flex shrink-0" onMouseDown={(event) => event.stopPropagation()}>{trailing}</span> : null}
        <BaseCombobox.Trigger tabIndex={-1} {...aria}
          className={clsx('flex h-full w-control shrink-0 items-center justify-center text-shell-muted outline-none', glyph)}>
          <ChevronDown />
        </BaseCombobox.Trigger>
      </BaseCombobox.InputGroup>
      <BaseCombobox.Portal container={usePopupLayer()}>
        <BaseCombobox.Positioner className="z-50 outline-none" sideOffset={4} align={inline ? 'end' : 'start'}>
          <BaseCombobox.Popup className={clsx(popupSurface, 'w-max min-w-[var(--anchor-width)] max-w-[var(--available-width)] data-[empty]:hidden')}>
            <BaseCombobox.List className="max-h-[min(22rem,var(--available-height))] overflow-y-auto">
              {(item: Option<Value>) => (
                <BaseCombobox.Item key={item.value} value={item} disabled={item.disabled}
                  className={clsx(popupItem, 'data-[selected]:font-medium')}>
                  <span className="flex w-[1.15em] justify-center">{indicator(item)}</span>
                  {item.glyph}
                  <span className={primaryText}>{item.label}</span>
                  {item.detail ? <span className={secondaryText}>{item.detail}</span> : null}
                </BaseCombobox.Item>
              )}
            </BaseCombobox.List>
          </BaseCombobox.Popup>
        </BaseCombobox.Positioner>
      </BaseCombobox.Portal>
    </BaseCombobox.Root>
  );
}
