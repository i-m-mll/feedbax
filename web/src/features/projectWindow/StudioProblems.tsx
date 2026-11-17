import { TriangleAlert, X } from 'lucide-react';
import { useId, useLayoutEffect, useRef, type ReactNode } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { commandAttributes, type StudioCommand } from './studioCommands';
import { Button, IconButton } from '@/components/ui/Button';

export function StudioProblems({ command, count, open, opener, onOpen, onClose, children }: {
  command: StudioCommand;
  count: number;
  open: boolean;
  opener: HTMLElement | null;
  onOpen: (opener: HTMLElement) => void;
  onClose: () => void;
  children: ReactNode;
}) {
  const id = useId();
  const anchor = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const dismiss = useRef(onClose);
  dismiss.current = onClose;

  function closeAndRestore() {
    dismiss.current();
    if (opener?.isConnected && opener.getClientRects().length && !opener.matches(':disabled') && !opener.closest('[inert], [hidden]')) {
      opener.focus();
    } else {
      trigger.current?.focus();
    }
  }

  useLayoutEffect(() => {
    const entry = anchor.current;
    const surface = panel.current;
    if (!open || !entry || !surface) return;
    const fit = () => {
      const rect = entry.getBoundingClientRect();
      surface.style.maxInlineSize = `${Math.max(0, rect.right - 8)}px`;
      surface.style.maxBlockSize = `${Math.max(0, rect.top - 16)}px`;
    };
    fit();
    const resize = new ResizeObserver(fit);
    resize.observe(entry);
    if (entry.parentElement) resize.observe(entry.parentElement);
    window.addEventListener('resize', fit);
    const outside = (event: PointerEvent) => {
      if (event.target instanceof Node && !entry.contains(event.target)) dismiss.current();
    };
    document.addEventListener('pointerdown', outside, true);
    close.current?.focus();
    return () => {
      resize.disconnect();
      window.removeEventListener('resize', fit);
      document.removeEventListener('pointerdown', outside, true);
    };
  }, [open, opener]);

  return (
    <div ref={anchor} className="relative"
      onBlur={(event) => {
        if (open && event.relatedTarget instanceof Node && !event.currentTarget.contains(event.relatedTarget)) dismiss.current();
      }}
      onKeyDown={(event) => {
        if (!open || event.key !== 'Escape') return;
        event.preventDefault();
        event.stopPropagation();
        closeAndRestore();
      }}>
      <Button ref={trigger} variant="ghost" {...commandAttributes(command)} aria-label={undefined} aria-haspopup="dialog"
        aria-expanded={open} aria-controls={open ? id : undefined}
        onClick={(event) => onOpen(event.currentTarget)} className="font-medium">
        <TriangleAlert aria-hidden="true" />
        {studioName('problems')} <span className="tabular-nums">{count}</span>
      </Button>
      {open && (
        <section ref={panel} id={id} role="dialog" aria-labelledby={`${id}-title`}
          className="absolute bottom-full right-0 z-50 mb-2 grid w-[34rem] gap-3 overflow-auto rounded-shell-control border border-shell-divider bg-shell-chrome p-4 text-shell-foreground shadow-xl">
          <header className="flex items-center justify-between gap-2">
            <h2 id={`${id}-title`} className="font-semibold">{studioName('problems')}</h2>
            <IconButton ref={close} label={studioName('close')} icon={<X aria-hidden="true" />} onClick={closeAndRestore} />
          </header>
          {children}
        </section>
      )}
    </div>
  );
}
