import clsx from 'clsx';
import { useEffect, type ReactNode } from 'react';

interface DialogProps {
  onClose: () => void;
  children: ReactNode;
  ariaLabel?: string;
  className?: string;
  panelClassName?: string;
  closeOnBackdrop?: boolean;
  closeOnEscape?: boolean;
}

export function Dialog({
  onClose,
  children,
  ariaLabel,
  className,
  panelClassName,
  closeOnBackdrop = true,
  closeOnEscape = true,
}: DialogProps) {
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
      <div className={panelClassName}>{children}</div>
    </div>
  );
}
