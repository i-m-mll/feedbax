import { useRef, type ReactElement } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { ContextMenu, ContextMenuContent, ContextMenuTrigger, MenuItem } from '@/components/ui/Menu';
import type { PlanRunView } from '@/features/pipeline/planView';
import type { StudioCommand } from '@/features/projectWindow/studioCommands';

/** The commands on a set of runs, with the element they were invoked from. */
export type RunMenu = (runs: readonly PlanRunView[], anchor: Element | undefined) => readonly StudioCommand[];

function RunMenuItems({ commands }: { commands: () => readonly StudioCommand[] }) {
  return commands().map((command) => (
    <MenuItem key={command.id} data-command-id={command.id} disabled={!command.enabled} onClick={command.invoke}>
      {studioName(command.name)}
    </MenuItem>
  ));
}

export function RunContextMenu({ runs, menu, trigger }: {
  runs: readonly PlanRunView[];
  menu?: RunMenu;
  trigger: ReactElement;
}) {
  const anchor = useRef<Element>();
  if (!menu) return trigger;
  return (
    <ContextMenu>
      <ContextMenuTrigger render={trigger} onContextMenu={(event) => { anchor.current = event.currentTarget; }} />
      <ContextMenuContent><RunMenuItems commands={() => menu(runs, anchor.current)} /></ContextMenuContent>
    </ContextMenu>
  );
}
