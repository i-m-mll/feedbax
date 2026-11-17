import { NumberField as BaseNumberField } from '@base-ui/react/number-field';
import clsx from 'clsx';
import { Minus, Plus } from 'lucide-react';
import { forwardRef } from 'react';
import { fieldBox, fieldInput, glyph } from './styles';

export interface NumberFieldProps {
  value: number | null;
  onValueChange: (value: number | null) => void;
  /** Fires when an edit ends: on blur after typing, on Enter, and on each step. */
  onValueCommitted?: (value: number | null) => void;
  min?: number;
  max?: number;
  step?: number;
  format?: Intl.NumberFormatOptions;
  disabled?: boolean;
  readOnly?: boolean;
  invalid?: boolean;
  placeholder?: string;
  'aria-label'?: string;
  id?: string;
  className?: string;
}

const exact: Intl.NumberFormatOptions = { useGrouping: false, maximumFractionDigits: 20 };

const stepper = clsx(
  'flex h-full w-control items-center justify-center text-shell-muted outline-none',
  'hover:[&:not([data-disabled])]:bg-shell-hover hover:text-shell-foreground data-[disabled]:opacity-45',
  glyph,
);

export const NumberField = forwardRef<HTMLInputElement, NumberFieldProps>(function NumberField(
  { value, onValueChange, onValueCommitted, min, max, step, format = exact, disabled, readOnly, invalid, placeholder, id, className, ...aria }, ref,
) {
  return (
    <BaseNumberField.Root
      id={id} value={value} onValueChange={(next) => onValueChange(next)}
      onValueCommitted={onValueCommitted && ((next) => onValueCommitted(next))}
      min={min} max={max} step={step} format={format} disabled={disabled} readOnly={readOnly}
    >
      <BaseNumberField.Group className={clsx(fieldBox, 'overflow-hidden', className)}>
        <BaseNumberField.Input
          ref={ref}
          aria-invalid={invalid || undefined}
          placeholder={placeholder}
          className={clsx(fieldInput, 'tabular-nums')}
          onKeyDown={(event) => { if (event.key === 'Enter' && !event.nativeEvent.isComposing) event.currentTarget.blur(); }}
          {...aria}
        />
        {readOnly ? null : <>
          <BaseNumberField.Decrement tabIndex={-1} className={clsx(stepper, 'border-l border-shell-divider')}><Minus /></BaseNumberField.Decrement>
          <BaseNumberField.Increment tabIndex={-1} className={clsx(stepper, 'border-l border-shell-divider')}><Plus /></BaseNumberField.Increment>
        </>}
      </BaseNumberField.Group>
    </BaseNumberField.Root>
  );
});
