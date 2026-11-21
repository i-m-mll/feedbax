import { useRef, type KeyboardEvent, type ReactElement, type SyntheticEvent } from 'react';
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

const stop = (event: SyntheticEvent) => event.stopPropagation();

/** The context menu of a thing: the commands that take it, opened by a right-click or the context menu keys. */
export function CommandContextMenu({ commands, trigger, focusAfter, onOpen }: {
  commands?: (anchor: Element | undefined) => readonly StudioCommand[];
  trigger: ReactElement;
  /** Called as the menu opens, before its commands are read: a thing outside the selection is selected first. */
  onOpen?: () => void;
  /** Where the chosen command wants the keyboard once the menu has closed; the open menu keeps it from going sooner. */
  focusAfter?: () => HTMLElement | null;
}) {
  const anchor = useRef<Element>();
  if (!commands) return trigger;
  return (
    <ContextMenu onOpenChange={(open) => { if (open) onOpen?.(); }} onOpenChangeComplete={(open) => {
      const target = open ? null : focusAfter?.();
      // The closed menu keeps the page inert until it unmounts, which follows this callback.
      if (target) requestAnimationFrame(() => target.focus());
    }}>
      <ContextMenuTrigger render={trigger} onKeyDown={contextMenuKey}
        onContextMenu={(event) => { anchor.current = event.currentTarget; }} />
      <ContextMenuContent>
        {/* React carries a portal's events to the trigger's ancestors; a canvas node would take the item's click. */}
        <div className="contents" onClick={stop} onPointerDown={stop} onMouseDown={stop} onDoubleClick={stop}>
          <CommandMenuItems commands={() => commands(anchor.current)} />
        </div>
      </ContextMenuContent>
    </ContextMenu>
  );
}
