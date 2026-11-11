import { Activity, X } from 'lucide-react';
import { useId, useRef } from 'react';
import { StudioActivityPane, useStudioActivityCount } from '@/components/activity/StudioActivity';
import { studioName } from '@/components/activity/studioVocabulary';
import type { StudioActivityJob } from '@/generated/studioContracts';

export function HeaderActivity({
  open,
  onToggle,
  onClose,
  onOpenRun,
  projectNames,
}: {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  onOpenRun: (job: StudioActivityJob) => Promise<boolean>;
  projectNames: ReadonlyMap<string, string>;
}) {
  const panelId = useId();
  const trigger = useRef<HTMLButtonElement | null>(null);
  const running = useStudioActivityCount();
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
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={onToggle}
        className="inline-flex items-center gap-1.5 rounded-md px-2 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-100"
      >
        <Activity aria-hidden="true" className="size-4" />
        {studioName('activity')}
        {running > 0 && <span className="tabular-nums">{running}</span>}
      </button>
      {open && (
        <section
          id={panelId}
          aria-label={studioName('activity')}
          onKeyDown={(event) => {
            if (event.key === 'Escape') {
              event.stopPropagation();
              closePanel();
            }
          }}
          className="absolute right-0 top-full z-50 mt-2 grid max-h-[calc(100vh-5rem)] w-[min(34rem,85vw)] gap-3 overflow-auto rounded-lg border border-slate-200 bg-white p-4 text-slate-900 shadow-xl"
        >
          <header className="flex items-center justify-between">
            <h2 className="font-semibold">{studioName('activity')}</h2>
            <button
              type="button"
              aria-label={studioName('close')}
              onClick={closePanel}
              className="rounded p-1 hover:bg-slate-100"
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
