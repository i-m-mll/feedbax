import { Button as BaseButton } from '@base-ui/react/button';
import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { forwardRef, type ComponentPropsWithoutRef, type ReactNode } from 'react';
import { disabledLook, focusRing, glyph } from './styles';

export type ButtonVariant = 'default' | 'primary' | 'ghost';

const variants: Record<ButtonVariant, string> = {
  default:
    'border border-shell-control bg-shell-surface hover:[&:not([data-disabled])]:bg-shell-hover active:[&:not([data-disabled])]:bg-shell-selected',
  primary:
    'border border-shell-foreground bg-shell-foreground font-medium text-shell-surface hover:[&:not([data-disabled])]:opacity-90 active:[&:not([data-disabled])]:opacity-80',
  ghost:
    'border border-transparent bg-transparent hover:[&:not([data-disabled])]:bg-shell-hover active:[&:not([data-disabled])]:bg-shell-selected',
};

const base = clsx(
  'inline-flex h-control shrink-0 select-none items-center justify-center gap-1.5 whitespace-nowrap rounded-control text-shell-foreground',
  focusRing, glyph, disabledLook,
);

type BaseProps = Omit<ComponentPropsWithoutRef<typeof BaseButton>, 'className'> & { className?: string };

export interface ButtonProps extends BaseProps {
  variant?: ButtonVariant;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'default', className, type = 'button', ...props }, ref,
) {
  return <BaseButton ref={ref} type={type} className={clsx(base, variants[variant], 'px-control-inline', className)} {...props} />;
});

export interface IconButtonProps extends Omit<ButtonProps, 'children' | 'aria-label'> {
  label: string;
  icon: ReactNode;
  pressed?: boolean;
  tipPlacement?: 'bottom' | 'right';
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(function IconButton(
  { label, icon, pressed, tipPlacement = 'bottom', variant = 'ghost', className, type = 'button', ...props }, ref,
) {
  return (
    <BaseButton
      ref={ref}
      type={type}
      aria-label={label}
      aria-pressed={pressed}
      focusableWhenDisabled
      className={clsx(base, variants[variant], 'w-control px-0 aria-pressed:bg-shell-selected', className)}
      {...appTip(label, tipPlacement)}
      {...props}
    >
      {icon}
    </BaseButton>
  );
});
