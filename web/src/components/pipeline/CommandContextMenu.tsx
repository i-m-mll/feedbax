import { useRef, type KeyboardEvent, type ReactElement } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import {
  ContextMenu, ContextMenuContent, ContextMenuTrigger, MenuItem, openContextMenu, opensContextMenu,
} from '@/components/ui/Menu';
import type { PlanRunView, PlanStepView } from '@/features/pipeline/planView';
import { shortcutLabel, type StudioCommand } from '@/features/projectWindow/studioCommands';

/** The commands on a set of runs, with the element they were invoked from. */
export type RunMenu = (runs: readonly PlanRunView[], anchor: Element | undefined) => readonly StudioCommand[];

/** The commands on a step; `rename` starts the edit of its name where the menu was opened. */
export type StepMenu = (step: PlanStepView, rename?: () => void) => readonly StudioCommand[];

/** A menu's items, read when the menu draws them. */
export function CommandMenuItems({ commands }: { commands: () => readonly StudioCommand[] }) {
  return commands().map((command) => (
    <MenuItem key={command.id} data-command-id={command.id} disabled={!command.enabled} onClick={command.invoke}
      shortcut={command.shortcut && shortcutLabel(command.shortcut)}>
      {studioName(command.name)}
    </MenuItem>
  ));
}

/** Opens the context menu of the focused element from the keyboard; the caller passes it the key down. */
export function contextMenuKey(event: KeyboardEvent) {
  if (!opensContextMenu(event)) return false;
  event.preventDefault();
  event.stopPropagation();
  openContextMenu(event.currentTarget);
  return true;
}

/** The context menu of a thing: the commands that take it, opened by a right-click or the context menu keys. */
export function CommandContextMenu({ commands, trigger }: {
  commands?: (anchor: Element | undefined) => readonly StudioCommand[];
  trigger: ReactElement;
}) {
  const anchor = useRef<Element>();
  if (!commands) return trigger;
  return (
    <ContextMenu>
      <ContextMenuTrigger render={trigger} onKeyDown={contextMenuKey}
        onContextMenu={(event) => { anchor.current = event.currentTarget; }} />
      <ContextMenuContent><CommandMenuItems commands={() => commands(anchor.current)} /></ContextMenuContent>
    </ContextMenu>
  );
}
