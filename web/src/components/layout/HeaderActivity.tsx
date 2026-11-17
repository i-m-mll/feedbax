import { Activity, X } from 'lucide-react';
import { useId, useLayoutEffect, useRef } from 'react';
import { StudioActivityPane, useStudioActivityCount } from '@/components/activity/StudioActivity';
import { studioName } from '@/components/activity/studioVocabulary';
import { commandAttributes, type StudioCommand } from '@/features/projectWindow/studioCommands';
import type { StudioActivityJob } from '@/generated/studioContracts';

export function HeaderActivity({
  open,
  onToggle,
  onClose,
  onOpenRun,
  projectNames,
  placement = 'below',
  command,
}: {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  onOpenRun: (job: StudioActivityJob) => Promise<boolean>;
  projectNames: ReadonlyMap<string, string>;
  placement?: 'above' | 'below';
  command?: StudioCommand;
}) {
  const panelId = useId();
  const trigger = useRef<HTMLButtonElement | null>(null);
  const running = useStudioActivityCount();
  const close = useRef<HTMLButtonElement>(null);
  const managed = !!command;
  useLayoutEffect(() => {
    if (open && managed) close.current?.focus();
  }, [open, managed]);
  const closePanel = () => {
    onClose();
    trigger.current?.focus();
  };

  return (
    <div className="relative">
      <button
        type="button"
        ref={trigger}
        aria-label={studioName('activity')}
        {...(command && commandAttributes(command))}
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={onToggle}
        className="inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium text-shell-muted hover:bg-shell-hover"
      >
        <Activity aria-hidden="true" className="size-4" />
        {studioName('activity')}
        {running > 0 && <span className="tabular-nums">{running}</span>}
      </button>
      {open && (
        <section
          id={panelId}
          data-studio-activity-panel={managed ? '' : undefined}
          aria-label={studioName('activity')}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              event.stopPropagation();
              closePanel();
            }
          }}
          className={`absolute right-0 z-50 grid max-h-[calc(100vh-5rem)] w-[min(34rem,85vw)] gap-3 overflow-auto rounded-lg border border-shell-divider bg-shell-chrome p-4 text-shell-foreground shadow-xl ${placement === 'above' ? 'bottom-full mb-2' : 'top-full mt-2'}`}
        >
          <header className="flex items-center justify-between">
            <h2 className="font-semibold">{studioName('activity')}</h2>
            <button
              type="button"
              ref={close}
              aria-label={studioName('close')}
              onClick={closePanel}
              className="rounded p-1 hover:bg-shell-hover"
            >
              <X aria-hidden="true" className="size-4" />
            </button>
          </header>
          <StudioActivityPane projectNames={projectNames} onClose={onClose} onOpenRun={onOpenRun} />
        </section>
      )}
    </div>
  );
}
