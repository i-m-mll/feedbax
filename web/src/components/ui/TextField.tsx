import { Input } from '@base-ui/react/input';
import clsx from 'clsx';
import { forwardRef, type ComponentPropsWithoutRef, type ReactNode } from 'react';
import { fieldBox, fieldInput, glyph } from './styles';

export interface TextFieldProps extends Omit<ComponentPropsWithoutRef<typeof Input>, 'className' | 'onValueChange'> {
  onValueChange?: (value: string) => void;
  invalid?: boolean;
  leading?: ReactNode;
  className?: string;
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(function TextField(
  { onValueChange, invalid, leading, className, ...props }, ref,
) {
  return (
    <div className={clsx(fieldBox, className)}>
      {leading ? <span className={clsx('flex pl-control-inline text-shell-muted', glyph)}>{leading}</span> : null}
      <Input
        ref={ref}
        aria-invalid={invalid || undefined}
        className={clsx(fieldInput, leading && 'pl-1.5')}
        onValueChange={onValueChange && ((value) => onValueChange(value))}
        {...props}
      />
    </div>
  );
});
