import clsx from 'clsx';
import { useEffect, useLayoutEffect, useRef, type ReactNode } from 'react';

interface DialogProps {
  onClose: () => void;
  children: ReactNode;
  ariaLabel?: string;
  className?: string;
  panelClassName?: string;
  closeOnBackdrop?: boolean;
  closeOnEscape?: boolean;
  manageFocus?: boolean;
}

export function Dialog({
  onClose,
  children,
  ariaLabel,
  className,
  panelClassName,
  closeOnBackdrop = true,
  closeOnEscape = true,
  manageFocus = false,
}: DialogProps) {
  const panel = useRef<HTMLDivElement>(null);
  function tabbable(container: HTMLElement) {
    return [...container.querySelectorAll<HTMLElement>(
      'button, input, select, textarea, a[href], summary, [tabindex]',
    )].filter((element) => element.tabIndex >= 0 && !element.matches(':disabled') &&
      !element.closest('[hidden], [inert]') && element.getClientRects().length > 0);
  }

  useLayoutEffect(() => {
    const container = panel.current;
    if (!manageFocus || !container) return;
    const previous = document.activeElement;
    (container.querySelector<HTMLElement>('[data-dialog-autofocus]') ?? tabbable(container)[0] ?? container).focus();
    return () => {
      if (previous instanceof HTMLElement && previous.isConnected &&
          (container.contains(document.activeElement) || document.activeElement === document.body)) previous.focus();
    };
  }, [manageFocus]);

  useEffect(() => {
    if (!closeOnEscape) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [closeOnEscape, onClose]);

  return (
    <div
      className={clsx(
        'z-50',
        className ?? 'fixed inset-0 flex items-center justify-center bg-black/30'
      )}
      onClick={(event) => {
        if (closeOnBackdrop && event.target === event.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
    >
      <div className={panelClassName} ref={panel} tabIndex={manageFocus ? -1 : undefined}
        onKeyDown={manageFocus ? (event) => {
          if (event.key !== 'Tab') return;
          const controls = tabbable(event.currentTarget);
          const boundary = event.shiftKey ? controls.at(-1) : controls[0];
          if (!boundary || event.target === event.currentTarget ||
              event.target === (event.shiftKey ? controls[0] : controls.at(-1))) {
            event.preventDefault();
            (boundary ?? event.currentTarget).focus();
          }
        } : undefined}>
        {children}
      </div>
    </div>
  );
}
