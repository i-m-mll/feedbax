import { Select as BaseSelect } from '@base-ui/react/select';
import clsx from 'clsx';
import { Check, ChevronsUpDown } from 'lucide-react';
import type { ReactNode } from 'react';
import { disabledLook, focusRing, glyph, leadingSlot, popupItem, popupSurface } from './styles';
import { usePopupLayer } from './popupLayer';

export interface Option<Value extends string> {
  value: Value;
  label: string;
  glyph?: ReactNode;
  /** Trailing text in the list, such as a declaration's package. */
  detail?: string;
  /** A mark after the label, in the list and in the trigger, such as a machine's state. */
  mark?: ReactNode;
  disabled?: boolean;
}

export interface SelectProps<Value extends string> {
  items: readonly Option<Value>[];
  value: NoInfer<Value> | null;
  onValueChange: (value: Value) => void;
  placeholder?: string;
  /** A glyph drawn at the start of the trigger and not in the list, such as the picker's kind. */
  leading?: ReactNode;
  disabled?: boolean;
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-describedby'?: string;
  id?: string;
  className?: string;
}

export function Select<Value extends string>(
  { items, value, onValueChange, placeholder, leading, disabled, id, className, ...aria }: SelectProps<Value>,
) {
  const selectedMark = items.find((item) => item.value === value)?.mark;
  return (
    <BaseSelect.Root
      items={items}
      value={value}
      onValueChange={(next) => { if (next !== null) onValueChange(next as Value); }}
      disabled={disabled}
    >
      <BaseSelect.Trigger
        id={id}
        className={clsx(
          'flex h-control min-w-0 select-none items-center gap-1 rounded-control border border-shell-field bg-shell-surface pr-control-inline text-shell-foreground',
          leading ? null : 'pl-control-inline',
          'hover:[&:not([data-disabled])]:bg-shell-hover data-[popup-open]:bg-shell-hover',
          focusRing, glyph, disabledLook, className,
        )}
        {...aria}
      >
        {leading ? <span aria-hidden="true" className={leadingSlot}>{leading}</span> : null}
        <BaseSelect.Value className="min-w-0 flex-1 truncate text-left data-[placeholder]:text-shell-muted" placeholder={placeholder} />
        {selectedMark}
        <BaseSelect.Icon className="flex text-shell-muted"><ChevronsUpDown /></BaseSelect.Icon>
      </BaseSelect.Trigger>
      <BaseSelect.Portal container={usePopupLayer()}>
        <BaseSelect.Positioner className="z-50 outline-none" sideOffset={4} alignItemWithTrigger={false} align="start">
          <BaseSelect.Popup className={clsx(popupSurface, 'min-w-[var(--anchor-width)]')}>
            <BaseSelect.List className="max-h-[var(--available-height)] overflow-y-auto">
              {items.map((item) => (
                <BaseSelect.Item key={item.value} value={item.value} disabled={item.disabled}
                  className={clsx(popupItem, 'pr-6 data-[selected]:font-medium')}>
                  <span className="flex w-[1.15em] justify-center">
                    <BaseSelect.ItemIndicator className="flex"><Check /></BaseSelect.ItemIndicator>
                  </span>
                  {item.glyph}
                  <BaseSelect.ItemText>{item.label}</BaseSelect.ItemText>
                  {item.mark}
                </BaseSelect.Item>
              ))}
            </BaseSelect.List>
          </BaseSelect.Popup>
        </BaseSelect.Positioner>
      </BaseSelect.Portal>
    </BaseSelect.Root>
  );
}
