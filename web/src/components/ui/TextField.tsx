import { Input } from '@base-ui/react/input';
import clsx from 'clsx';
import { forwardRef, type ComponentPropsWithoutRef, type ReactNode, type TextareaHTMLAttributes } from 'react';
import { fieldBox, fieldFrame, fieldInput, glyph, leadingSlot } from './styles';

export interface TextFieldProps extends Omit<ComponentPropsWithoutRef<typeof Input>, 'className' | 'onValueChange'> {
  onValueChange?: (value: string) => void;
  invalid?: boolean;
  leading?: ReactNode;
  trailing?: ReactNode;
  /** A control sharing the box, at its end past a hairline: a class's picker. */
  end?: ReactNode;
  className?: string;
}

export const TextField = forwardRef<HTMLInputElement, TextFieldProps>(function TextField(
  { onValueChange, invalid, leading, trailing, end, className, ...props }, ref,
) {
  return (
    <div data-field-box="" className={clsx(fieldBox, className)}>
      {leading ? <span className={leadingSlot}>{leading}</span> : null}
      <Input
        ref={ref}
        aria-invalid={invalid || undefined}
        className={clsx(fieldInput, leading && 'pl-1')}
        onValueChange={onValueChange && ((value) => onValueChange(value))}
        {...props}
      />
      {trailing ? <span className={clsx('flex pr-control-inline text-shell-muted', glyph)}>{trailing}</span> : null}
      {end ? <span className="field-end flex h-full min-w-control shrink-[1000] border-l border-shell-field">{end}</span> : null}
    </div>
  );
});

export interface TextAreaProps extends Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'className'> {
  invalid?: boolean;
  className?: string;
}

export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(function TextArea(
  { invalid, className, ...props }, ref,
) {
  return (
    <div className={clsx(fieldFrame, className)}>
      <textarea
        ref={ref}
        aria-invalid={invalid || undefined}
        className={clsx(fieldInput, 'min-h-[calc(3*var(--row-height))] resize-y py-1')}
        {...props}
      />
    </div>
  );
});
