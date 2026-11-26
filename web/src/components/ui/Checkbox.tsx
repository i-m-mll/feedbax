import { Checkbox as BaseCheckbox } from '@base-ui/react/checkbox';
import clsx from 'clsx';
import { Check, Minus } from 'lucide-react';
import type { ReactNode } from 'react';
import { focusRing } from './styles';

export interface CheckboxProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  indeterminate?: boolean;
  disabled?: boolean;
  label?: ReactNode;
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-describedby'?: string;
  'aria-invalid'?: boolean;
  className?: string;
}

export function Checkbox({ checked, onCheckedChange, indeterminate, disabled, label, className, ...aria }: CheckboxProps) {
  const box = (
    <BaseCheckbox.Root
      checked={checked}
      indeterminate={indeterminate}
      disabled={disabled}
      onCheckedChange={(next) => onCheckedChange(next)}
      className={clsx(
        'flex size-[1.15em] shrink-0 items-center justify-center rounded-row border border-shell-control bg-shell-surface text-shell-surface',
        'data-[checked]:border-shell-foreground data-[checked]:bg-shell-foreground',
        'data-[indeterminate]:border-shell-foreground data-[indeterminate]:bg-shell-foreground',
        focusRing, !label && clsx('data-[disabled]:opacity-45', className),
      )}
      {...aria}
    >
      <BaseCheckbox.Indicator className="flex [&_svg]:size-[0.9em]" keepMounted={false}>
        {indeterminate ? <Minus strokeWidth={3} /> : <Check strokeWidth={3} />}
      </BaseCheckbox.Indicator>
    </BaseCheckbox.Root>
  );
  if (!label) return box;
  return (
    <label className={clsx('inline-flex h-control cursor-default select-none items-center gap-2 text-shell-foreground has-[[data-disabled]]:opacity-45', className)}>
      {box}
      <span>{label}</span>
    </label>
  );
}
