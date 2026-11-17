import { Button as BaseButton } from '@base-ui/react/button';
import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { forwardRef, type ComponentPropsWithoutRef, type ReactNode } from 'react';
import { disabledLook, focusRing, glyph } from './styles';

export type ButtonVariant = 'default' | 'primary' | 'ghost';
export type ButtonSize = 'control' | 'mark';
export type ButtonAlign = 'center' | 'start';

const variants: Record<ButtonVariant, string> = {
  default:
    'border border-shell-control bg-shell-surface hover:[&:not([data-disabled])]:bg-shell-hover active:[&:not([data-disabled])]:bg-shell-selected',
  primary:
    'border border-shell-foreground bg-shell-foreground font-medium text-shell-surface hover:[&:not([data-disabled])]:opacity-90 active:[&:not([data-disabled])]:opacity-80',
  ghost:
    'border border-transparent bg-transparent hover:[&:not([data-disabled])]:bg-shell-hover active:[&:not([data-disabled])]:bg-shell-selected',
};

const textSizes: Record<ButtonSize, string> = { control: 'h-control px-control-inline', mark: 'h-[1.5em] px-1 text-xs' };
const alignments: Record<ButtonAlign, string> = { center: 'justify-center', start: 'justify-start text-left' };
const iconSizes: Record<ButtonSize, string> = { control: 'h-control w-control', mark: 'size-[1.5em] text-xs' };

const base = clsx(
  'inline-flex shrink-0 select-none items-center gap-1.5 whitespace-nowrap rounded-control text-shell-foreground',
  'aria-pressed:bg-shell-selected aria-pressed:font-medium',
  focusRing, glyph, disabledLook,
);

type BaseProps = Omit<ComponentPropsWithoutRef<typeof BaseButton>, 'className'> & { className?: string };

export interface ButtonProps extends BaseProps {
  variant?: ButtonVariant;
  size?: ButtonSize;
  align?: ButtonAlign;
  pressed?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'default', size = 'control', align = 'center', pressed, className, type = 'button', ...props }, ref,
) {
  return <BaseButton ref={ref} type={type} aria-pressed={pressed}
    className={clsx(base, variants[variant], textSizes[size], alignments[align], className)} {...props} />;
});

export interface IconButtonProps extends Omit<ButtonProps, 'children' | 'aria-label' | 'align'> {
  label: string;
  icon: ReactNode;
  pressed?: boolean;
  tipPlacement?: 'bottom' | 'right';
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, pressed, tipPlacement = 'bottom', variant = 'ghost', size = 'control', className, type = 'button', ...props }, ref,
) {
  return (
    <BaseButton
      ref={ref}
      type={type}
      aria-label={label}
      aria-pressed={pressed}
      focusableWhenDisabled
      className={clsx(base, variants[variant], iconSizes[size], 'justify-center px-0', className)}
      {...appTip(label, tipPlacement)}
      {...props}
    >
      {icon}
    </BaseButton>
  );
});
