import { NumberField as BaseNumberField } from '@base-ui/react/number-field';
import clsx from 'clsx';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { forwardRef } from 'react';
import { fieldBox, fieldInput } from './styles';

export interface NumberFieldProps {
  value: number | null;
  onValueChange: (value: number | null) => void;
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

const stepper = clsx(
  'flex min-h-0 flex-1 items-center justify-center text-shell-muted outline-none',
  'hover:[&:not([data-disabled])]:bg-shell-hover hover:text-shell-foreground data-[disabled]:opacity-45',
  '[&_svg]:size-3 [&_svg]:shrink-0',
);

export const NumberField = forwardRef<HTMLInputElement, NumberFieldProps>(function NumberField(
  { value, onValueChange, min, max, step, format, disabled, readOnly, invalid, placeholder, id, className, ...aria }, ref,
) {
  return (
    <BaseNumberField.Root
      id={id} value={value} onValueChange={(next) => onValueChange(next)}
      min={min} max={max} step={step} format={format} disabled={disabled} readOnly={readOnly}
    >
      <BaseNumberField.Group className={clsx(fieldBox, 'overflow-hidden', className)}>
        <BaseNumberField.Input
          ref={ref}
          aria-invalid={invalid || undefined}
          placeholder={placeholder}
          className={clsx(fieldInput, 'tabular-nums')}
          {...aria}
        />
        {readOnly ? null : (
          <span className="flex h-full w-4 shrink-0 flex-col border-l border-shell-divider">
            <BaseNumberField.Increment tabIndex={-1} className={stepper}><ChevronUp /></BaseNumberField.Increment>
            <BaseNumberField.Decrement tabIndex={-1} className={stepper}><ChevronDown /></BaseNumberField.Decrement>
          </span>
        )}
      </BaseNumberField.Group>
    </BaseNumberField.Root>
  );
});
