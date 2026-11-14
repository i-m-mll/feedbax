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
  const dialog = useRef<HTMLDialogElement>(null);
  function tabbable(container: HTMLElement) {
    return [...container.querySelectorAll<HTMLElement>(
      'button, input, select, textarea, a[href], summary, [tabindex]',
    )].filter((element) => element.tabIndex >= 0 && !element.matches(':disabled') &&
      !element.closest('[hidden], [inert]') && element.getClientRects().length > 0);
  }

  useLayoutEffect(() => {
    const container = panel.current;
    const modal = dialog.current;
    if (!manageFocus || !container || !modal) return;
    const previous = document.activeElement;
    const returnTargets: HTMLElement[] = [];
    for (let candidate = previous; candidate; candidate = candidate.parentElement) {
      if (candidate instanceof HTMLElement && (candidate === previous || candidate.hasAttribute('tabindex'))) {
        returnTargets.push(candidate);
      }
    }
    modal.showModal();
    (container.querySelector<HTMLElement>('[data-dialog-autofocus]') ?? tabbable(container)[0] ?? container).focus();
    return () => {
      const restore = container.contains(document.activeElement) || document.activeElement === document.body;
      modal.close();
      if (!restore) return;
      returnTargets.find((target) => target.isConnected && target.getClientRects().length > 0 &&
        !target.matches(':disabled') && !target.closest('[hidden], [inert]'))?.focus();
    };
  }, [manageFocus]);

  useEffect(() => {
    if (!closeOnEscape) return undefined;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (manageFocus) event.preventDefault();
      onClose();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [closeOnEscape, manageFocus, onClose]);

  return (
    <dialog
      ref={dialog}
      open={!manageFocus}
      className={clsx(
        'z-50 m-0 h-full max-h-none w-full max-w-none border-0 p-0 text-inherit backdrop:bg-transparent',
        className ?? 'fixed inset-0 flex items-center justify-center bg-black/30',
        className && 'bg-transparent',
      )}
      onClick={(event) => {
        if (closeOnBackdrop && event.target === event.currentTarget) onClose();
      }}
      role="dialog"
      aria-modal="true"
      aria-label={ariaLabel}
      onCancel={manageFocus ? (event) => event.preventDefault() : undefined}
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
    </dialog>
  );
}
