import { Activity, X } from 'lucide-react';
import { useId, useLayoutEffect, useRef } from 'react';
import { StudioActivityPane, useStudioActivityCount, type ActivityResultState, type ActivityRunName } from '@/components/activity/StudioActivity';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { commandTip, type StudioCommand } from '@/features/projectWindow/studioCommands';
import type { StudioActivityJob } from '@/generated/studioContracts';
import { IconButton } from '@/components/ui/Button';

export function HeaderActivity({
  open,
  onToggle,
  onClose,
  onOpenRun,
  projectNames,
  placement = 'below',
  command,
  resultState,
  runName,
}: {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  onOpenRun: (job: StudioActivityJob) => Promise<boolean>;
  projectNames: ReadonlyMap<string, string>;
  placement?: 'above' | 'below';
  command?: StudioCommand;
  resultState?: ActivityResultState;
  runName?: ActivityRunName;
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
      <IconButton
        ref={trigger}
        {...(command && commandTip(command))}
        label={studioAccessibleName('activity', running || undefined)}
        icon={<Activity aria-hidden="true" />}
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={onToggle}
      />
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
            <IconButton ref={close} label={studioName('close')} icon={<X aria-hidden="true" />} onClick={closePanel} />
          </header>
          <StudioActivityPane projectNames={projectNames} onClose={onClose} onOpenRun={onOpenRun} resultState={resultState} runName={runName} />
        </section>
      )}
    </div>
  );
}
