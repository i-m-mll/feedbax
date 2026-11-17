import { Toggle } from '@base-ui/react/toggle';
import { ToggleGroup } from '@base-ui/react/toggle-group';
import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import type { ReactNode } from 'react';
import { disabledLook, focusRing, glyph } from './styles';

export interface Segment<Value extends string> {
  value: Value;
  label: string;
  icon?: ReactNode;
  disabled?: boolean;
}

export interface SegmentedControlProps<Value extends string> {
  items: readonly Segment<Value>[];
  value: NoInfer<Value>;
  onValueChange: (value: Value) => void;
  'aria-label': string;
  disabled?: boolean;
  className?: string;
}

export function SegmentedControl<Value extends string>(
  { items, value, onValueChange, disabled, className, ...aria }: SegmentedControlProps<Value>,
) {
  return (
    <ToggleGroup
      value={[value]}
      onValueChange={(next) => { if (next.length) onValueChange(next[0] as Value); }}
      disabled={disabled}
      className={clsx('inline-flex h-control items-stretch gap-px rounded-control bg-shell-subtle p-px', className)}
      {...aria}
    >
      {items.map((item) => (
        <Toggle
          key={item.value}
          value={item.value}
          disabled={item.disabled}
          aria-label={item.icon ? item.label : undefined}
          {...(item.icon ? appTip(item.label) : {})}
          className={clsx(
            'inline-flex select-none items-center justify-center gap-1.5 rounded-row text-shell-muted',
            item.icon ? 'aspect-square' : 'px-control-inline',
            'hover:[&:not([data-disabled])]:text-shell-foreground',
            'data-[pressed]:bg-shell-surface data-[pressed]:font-medium data-[pressed]:text-shell-foreground',
            focusRing, glyph, disabledLook,
          )}
        >
          {item.icon ?? item.label}
        </Toggle>
      ))}
    </ToggleGroup>
  );
}
