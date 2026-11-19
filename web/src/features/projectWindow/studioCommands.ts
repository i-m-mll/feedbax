import { appTip } from '@window-shell/browser/tooltip';
import {
  ArrowLeft, ArrowRight, Command, Grid2x2, OctagonX, PanelLeft, PanelRight, Play, Square, SquareArrowOutUpRight, StepForward,
  type LucideIcon,
} from 'lucide-react';
import { studioAccessibleName, studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import type { AppearanceCommandId } from '@/appearance/appearance';
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
  settings: primary(','),
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
  modelSelectAll: primary('A'),
  modelRemove: { key: 'Backspace' },
  renameStep: { key: 'F2' },
  modelRename: { key: 'F2' },
  modelDuplicate: primary('D'),
  modelGroup: primary('G'),
  modelUngroup: primary('G', { shift: true }),
  modelCopy: primary('C'),
  modelCut: primary('X'),
  modelPaste: primary('V'),
  leave: primary('ArrowUp'),
} satisfies Partial<Record<StudioNameId, Shortcut>>;

const glyphs: Partial<Record<StudioCommandId, LucideIcon>> = {
  back: ArrowLeft,
  forward: ArrowRight,
  showSteps: PanelLeft,
  commandPalette: Command,
  showDrawer: PanelLeft,
  showInspector: PanelRight,
  run: Play,
  stop: Square,
  forceStop: OctagonX,
  resume: StepForward,
  openRun: SquareArrowOutUpRight,
  pivot: Grid2x2,
};

export type StudioCommandId = keyof typeof shortcuts | ProjectWindowView | AppearanceCommandId |
  'check' | 'accept' | 'newStep' | 'newSweep' | 'openRun' | 'resume' | 'forceStop' | 'activity' | 'problems' | 'retry' |
  'pivot' | 'dismiss' | 'connectPorts' | 'exposeOutput' |
  'flip' | 'enter' | 'collapseAll' | 'lock' | 'addDelay' | 'addIntervention' | 'addProbe' | 'addWaypoint' | 'removeWaypoint' |
  'addText' | 'addFigure' | 'moveEarlier' | 'moveLater' | 'remove' |
  'fitView' | 'zoomIn' | 'zoomOut' | 'rename';

const selectionCommands = new Set<StudioCommandId>([
  'run', 'stop', 'forceStop', 'resume', 'openRun', 'check', 'accept', 'newStep', 'newSweep', 'pivot',
  'addComponent', 'connectPorts', 'exposeOutput', 'addText', 'addFigure', 'moveEarlier', 'moveLater', 'remove',
  'flip', 'enter', 'leave', 'collapseAll', 'lock', 'addDelay', 'addIntervention', 'addProbe', 'addWaypoint', 'removeWaypoint',
  'renameStep', 'modelSelectAll', 'modelRemove', 'modelRename', 'modelDuplicate', 'modelGroup', 'modelUngroup',
  'modelCopy', 'modelCut', 'modelPaste',
]);

export interface StudioCommand {
  readonly id: StudioCommandId;
  readonly name: StudioNameId;
  /** The visible name. */
  readonly title: string;
  /** The accessible name, with the command's subject. */
  readonly label: string;
  readonly glyph?: LucideIcon;
  readonly enabled: boolean;
  readonly selectionScoped: boolean;
  readonly shortcut?: Shortcut;
  invoke(): void;
}

export function bindStudioCommand<Id extends StudioCommandId>(
  id: Id,
  perform: () => void,
  { enabled = true, name = id, subject, keyboard = true, glyph = glyphs[id] }:
    { enabled?: boolean; name?: StudioNameId; subject?: string; keyboard?: boolean; glyph?: LucideIcon } = {},
): StudioCommand & { readonly id: Id } {
  const viewPosition = projectWindowViews.indexOf(id as ProjectWindowView) + 1;
  const shortcut = (shortcuts as Partial<Record<StudioCommandId, Shortcut>>)[id] ??
    (viewPosition > 0 && viewPosition <= 9 ? primary(String(viewPosition)) : undefined);
  return {
    id, name, enabled, selectionScoped: selectionCommands.has(id), shortcut: keyboard ? shortcut : undefined,
    title: studioName(name), label: studioAccessibleName(name, subject), glyph,
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
  Backslash: '\\', Comma: ',', BracketLeft: '[', BracketRight: ']', Period: '.',
};

export function matchesShortcut(event: KeyboardEvent, shortcut: Shortcut | undefined) {
  if (!shortcut) return false;
  const key = punctuation[event.code] ?? (/^Digit[0-9]$/.test(event.code) ? event.code.at(-1) : event.key);
  return key?.toLowerCase() === shortcut.key.toLowerCase() &&
    (event.metaKey || event.ctrlKey) === Boolean(shortcut.primary) &&
    (event.metaKey && event.ctrlKey) === Boolean(shortcut.control) &&
    event.altKey === Boolean(shortcut.alt) && event.shiftKey === Boolean(shortcut.shift);
}

/**
 * Whether a control took the key for itself. Base UI cancels every key but Tab on a disabled control that keeps
 * focus, before any handler of ours runs; that cancel does not mean the control used the key.
 */
export function keyHandled(event: KeyboardEvent) {
  if (!event.defaultPrevented) return false;
  return !(event.target instanceof Element && event.target.matches('[aria-disabled="true"][data-disabled]'));
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

/** Commands that act on a canvas's selection; a focused text field keeps their keys. */
const canvasCommands = new Set<StudioCommandId>([
  'renameStep', 'addComponent', 'leave', 'modelSelectAll', 'modelRemove', 'modelRename', 'modelDuplicate', 'modelGroup',
  'modelUngroup', 'modelCopy', 'modelCut', 'modelPaste',
]);

/** ⌘Z, ⇧⌘Z, ⌘A, ⌘C, ⌘V, ⌘X and ⌘-arrows edit or move within text. */
function editsText(shortcut: Shortcut | undefined) {
  return !!shortcut?.primary && !shortcut.control && !shortcut.alt &&
    (['z', 'a', 'c', 'v', 'x'].includes(shortcut.key.toLowerCase()) || shortcut.key.startsWith('Arrow'));
}

/** Inside a text field, window commands still run from their chords; text-editing chords and canvas commands do not. */
export function commandInContext(command: StudioCommand, context: ReturnType<typeof studioCommandContext>) {
  if (context.dialog && !(context.paletteOpen && command.id === 'commandPalette')) return false;
  if (context.editable && (canvasCommands.has(command.id) ||
      editsText((shortcuts as Partial<Record<StudioCommandId, Shortcut>>)[command.id]))) return false;
  return !(context.activity && command.selectionScoped);
}

const jobAction = { stop: 'checkpoint', forceStop: 'force', resume: 'resume' } as const;

export function jobAllowsCommand(job: { allowed_actions: readonly string[] }, command: keyof typeof jobAction) {
  return job.allowed_actions.includes(jobAction[command]);
}
