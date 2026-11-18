import { Combobox as BaseCombobox } from '@base-ui/react/combobox';
import clsx from 'clsx';
import { Check, ChevronDown } from 'lucide-react';
import { useMemo } from 'react';
import type { Option } from './Select';
import { fieldBox, fieldInput, glyph, popupItem, popupSurface } from './styles';
import { usePopupLayer } from './popupLayer';

export interface ComboboxProps<Value extends string> {
  items: readonly Option<Value>[];
  value: NoInfer<Value> | null;
  onValueChange: (value: Value | null) => void;
  placeholder?: string;
  disabled?: boolean;
  invalid?: boolean;
  'aria-label'?: string;
  'aria-labelledby'?: string;
  id?: string;
  className?: string;
}

export function Combobox<Value extends string>(
  { items, value, onValueChange, placeholder, disabled, invalid, id, className, ...aria }: ComboboxProps<Value>,
) {
  const selected = useMemo(() => items.find((item) => item.value === value) ?? null, [items, value]);
  return (
    <BaseCombobox.Root
      items={items}
      value={selected}
      onValueChange={(next: Option<Value> | null) => onValueChange(next?.value ?? null)}
      isItemEqualToValue={(item: Option<Value>, current: Option<Value>) => item.value === current.value}
      disabled={disabled}
      autoHighlight
    >
      <BaseCombobox.InputGroup className={clsx(fieldBox, 'overflow-hidden', className)}>
        <BaseCombobox.Input id={id} placeholder={placeholder} aria-invalid={invalid || undefined}
          className={fieldInput} {...aria} />
        <BaseCombobox.Trigger tabIndex={-1} {...aria}
          className={clsx('flex h-full w-control items-center justify-center text-shell-muted outline-none', glyph)}>
          <ChevronDown />
        </BaseCombobox.Trigger>
      </BaseCombobox.InputGroup>
      <BaseCombobox.Portal container={usePopupLayer()}>
        <BaseCombobox.Positioner className="z-50 outline-none" sideOffset={4} align="start">
          <BaseCombobox.Popup className={clsx(popupSurface, 'w-[var(--anchor-width)] max-w-[var(--available-width)] data-[empty]:hidden')}>
            <BaseCombobox.List className="max-h-[min(22rem,var(--available-height))] overflow-y-auto">
              {(item: Option<Value>) => (
                <BaseCombobox.Item key={item.value} value={item} disabled={item.disabled}
                  className={clsx(popupItem, 'data-[selected]:font-medium')}>
                  <span className="flex w-[1.15em] justify-center">
                    <BaseCombobox.ItemIndicator className="flex"><Check /></BaseCombobox.ItemIndicator>
                  </span>
                  {item.glyph}
                  <span className="flex-1 truncate">{item.label}</span>
                  {item.detail ? <span className="truncate pl-4 text-shell-muted">{item.detail}</span> : null}
                </BaseCombobox.Item>
              )}
            </BaseCombobox.List>
          </BaseCombobox.Popup>
        </BaseCombobox.Positioner>
      </BaseCombobox.Portal>
    </BaseCombobox.Root>
  );
}
