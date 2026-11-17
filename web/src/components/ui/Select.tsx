import { Select as BaseSelect } from '@base-ui/react/select';
import clsx from 'clsx';
import { Check, ChevronsUpDown } from 'lucide-react';
import type { ReactNode } from 'react';
import { disabledLook, focusRing, glyph, popupItem, popupSurface } from './styles';

export interface Option<Value extends string> {
  value: Value;
  label: string;
  glyph?: ReactNode;
  disabled?: boolean;
}

export interface SelectProps<Value extends string> {
  items: readonly Option<Value>[];
  value: NoInfer<Value> | null;
  onValueChange: (value: Value) => void;
  placeholder?: string;
  disabled?: boolean;
  'aria-label'?: string;
  id?: string;
  className?: string;
}

export function Select<Value extends string>(
  { items, value, onValueChange, placeholder, disabled, id, className, ...aria }: SelectProps<Value>,
) {
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
          'flex h-control min-w-0 select-none items-center justify-between gap-2 rounded-control border border-shell-control bg-shell-surface px-control-inline text-shell-foreground',
          'hover:[&:not([data-disabled])]:bg-shell-hover data-[popup-open]:bg-shell-hover',
          focusRing, glyph, disabledLook, className,
        )}
        {...aria}
      >
        <BaseSelect.Value className="truncate data-[placeholder]:text-shell-muted" placeholder={placeholder} />
        <BaseSelect.Icon className="flex text-shell-muted"><ChevronsUpDown /></BaseSelect.Icon>
      </BaseSelect.Trigger>
      <BaseSelect.Portal>
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
                </BaseSelect.Item>
              ))}
            </BaseSelect.List>
          </BaseSelect.Popup>
        </BaseSelect.Positioner>
      </BaseSelect.Portal>
    </BaseSelect.Root>
  );
}
