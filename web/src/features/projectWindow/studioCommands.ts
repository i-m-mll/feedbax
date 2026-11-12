import { appTip } from '@window-shell/browser/tooltip';
import { studioAccessibleName, studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import { projectWindowViews, type ProjectWindowView } from './projectWindowState';

interface Shortcut {
  readonly key: string;
  readonly primary?: true;
  readonly control?: true;
  readonly alt?: true;
  readonly shift?: true;
}

const primary = (key: string, modifiers: Omit<Shortcut, 'key' | 'primary'> = {}): Shortcut => ({
  key, primary: true, ...modifiers,
});

const shortcuts = {
  commandPalette: primary('K'),
  undo: primary('Z'),
  redo: primary('Z', { shift: true }),
  run: primary('R'),
  stop: primary('.'),
  showMap: primary('1', { control: true }),
  showBoth: primary('2', { control: true }),
  showDetails: primary('3', { control: true }),
  showSteps: primary('S', { control: true }),
  showDrawer: primary('0'),
  showInspector: primary('0', { alt: true }),
  splitDetails: primary('\\'),
  switchSlot: primary('\\', { alt: true }),
  back: primary('['),
  forward: primary(']'),
  nextRegion: { key: 'F6' },
  previousRegion: { key: 'F6', shift: true },
  clearFocus: { key: 'Escape' },
  addComponent: { key: 'A', shift: true },
} satisfies Partial<Record<StudioNameId, Shortcut>>;

export type StudioCommandId = keyof typeof shortcuts | ProjectWindowView |
  'newStep' | 'newSweep' | 'openRun' | 'resume' | 'forceStop' | 'activity' | 'problems' | 'retry' |
  'pivot' | 'fold' | 'unfold' | 'dismiss' | 'connectPorts' | 'exposeOutput';

const selectionCommands = new Set<StudioCommandId>([
  'run', 'stop', 'forceStop', 'resume', 'openRun', 'newStep', 'newSweep', 'fold', 'unfold', 'pivot',
  'addComponent', 'connectPorts', 'exposeOutput',
]);

export interface StudioCommand {
  readonly id: StudioCommandId;
  readonly name: StudioNameId;
  readonly label: string;
  readonly enabled: boolean;
  readonly selectionScoped: boolean;
  readonly shortcut?: Shortcut;
  invoke(): void;
}

export function bindStudioCommand(
  id: StudioCommandId,
  perform: () => void,
  { enabled = true, name = id, subject, keyboard = true }:
    { enabled?: boolean; name?: StudioNameId; subject?: string; keyboard?: boolean } = {},
): StudioCommand {
  const viewPosition = projectWindowViews.indexOf(id as ProjectWindowView) + 1;
  const shortcut = (shortcuts as Partial<Record<StudioCommandId, Shortcut>>)[id] ??
    (viewPosition > 0 && viewPosition <= 9 ? primary(String(viewPosition)) : undefined);
  return {
    id, name, enabled, selectionScoped: selectionCommands.has(id), shortcut: keyboard ? shortcut : undefined,
    label: studioAccessibleName(name, subject),
    invoke: () => { if (enabled) perform(); },
  };
}

export function shortcutLabel(shortcut: Shortcut) {
  return [
    shortcut.control && '⌃', shortcut.alt && '⌥', shortcut.shift && '⇧', shortcut.primary && '⌘', shortcut.key,
  ].filter(Boolean).join('');
}

function ariaShortcut(shortcut: Shortcut) {
  return [
    shortcut.control && 'Control', shortcut.alt && 'Alt', shortcut.shift && 'Shift', shortcut.primary && 'Meta', shortcut.key,
  ].filter(Boolean).join('+');
}

export function commandAttributes(command: StudioCommand, placement?: 'bottom' | 'right') {
  const label = studioName(command.name);
  return {
    'data-command-id': command.id,
    'aria-label': command.label,
    'aria-keyshortcuts': command.shortcut && ariaShortcut(command.shortcut),
    ...appTip(command.shortcut ? `${label} · ${shortcutLabel(command.shortcut)}` : label, placement),
  };
}

const punctuation: Record<string, string> = {
  Backslash: '\\', BracketLeft: '[', BracketRight: ']', Period: '.',
};

export function matchesShortcut(event: KeyboardEvent, shortcut: Shortcut | undefined) {
  if (!shortcut) return false;
  const key = punctuation[event.code] ?? (/^Digit[0-9]$/.test(event.code) ? event.code.at(-1) : event.key);
  return key?.toLowerCase() === shortcut.key.toLowerCase() &&
    (event.metaKey || event.ctrlKey) === Boolean(shortcut.primary) &&
    (event.metaKey && event.ctrlKey) === Boolean(shortcut.control) &&
    event.altKey === Boolean(shortcut.alt) && event.shiftKey === Boolean(shortcut.shift);
}

export function studioCommandContext(target: EventTarget | null, paletteOpen: boolean) {
  const element = target instanceof Element ? target : null;
  return {
    paletteOpen,
    dialog: !!element?.closest('[role="dialog"]'),
    editable: !!element?.closest('input, textarea, select, [contenteditable="true"]'),
    activity: !!element?.closest('[data-studio-activity-panel]'),
  };
}

export function commandInContext(command: StudioCommand, context: ReturnType<typeof studioCommandContext>) {
  if (context.dialog && !(context.paletteOpen && command.id === 'commandPalette')) return false;
  if (context.editable && (command.id === 'undo' || command.id === 'redo' || command.id === 'run' || command.id === 'addComponent')) return false;
  return !(context.activity && command.selectionScoped);
}

const jobAction = { stop: 'checkpoint', forceStop: 'force', resume: 'resume' } as const;

export function jobAllowsCommand(job: { allowed_actions: readonly string[] }, command: keyof typeof jobAction) {
  return job.allowed_actions.includes(jobAction[command]);
}
