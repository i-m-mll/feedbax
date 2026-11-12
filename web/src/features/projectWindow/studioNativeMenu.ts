import { useLayoutEffect, useRef } from 'react';
import { projectWindowViews } from './projectWindowState';
import {
  bindStudioCommand, commandInContext, studioCommandContext, type StudioCommand, type StudioCommandId,
} from './studioCommands';

const groups = {
  Edit: ['undo', 'redo'],
  View: [
    'commandPalette', 'showMap', 'showBoth', 'showDetails', 'showSteps', 'showDrawer', 'showInspector',
    'splitDetails', 'switchSlot', ...projectWindowViews, 'back', 'forward', 'nextRegion', 'previousRegion',
    'clearFocus', 'activity', 'problems', 'retry',
  ],
  Run: ['run', 'stop', 'resume', 'forceStop', 'openRun'],
  Plan: ['newStep', 'newSweep', 'pivot', 'fold'],
} as const satisfies Record<string, readonly StudioCommandId[]>;
const sections = Object.entries(groups) as [keyof typeof groups, readonly StudioCommandId[]][];
const menuIds = new Set<StudioCommandId>([...sections.flatMap(([, ids]) => ids), 'unfold']);
const nativeCommandEvent = 'studio-native-command';
type Context = ReturnType<typeof studioCommandContext>;

function accelerator(command: StudioCommand) {
  const shortcut = command.shortcut;
  // Escape also cancels controls and dialogs inside the webview.
  if (!shortcut || command.id === 'clearFocus') return undefined;
  return [
    shortcut.control && 'Ctrl', shortcut.alt && 'Alt', shortcut.shift && 'Shift',
    shortcut.primary && 'Cmd', shortcut.key,
  ].filter(Boolean).join('+');
}

export function nativeOwnsShortcut(command: StudioCommand, event: KeyboardEvent) {
  return !!window.__FEEDBAX_STUDIO_MENU__ && menuIds.has(command.id) && !!accelerator(command) &&
    (!command.shortcut?.primary || event.metaKey);
}

export function studioMenuPresentation(commands: readonly StudioCommand[], context: Context, composing = false) {
  const bindings = new Map(commands.map((command) => [command.id, command]));
  return {
    textEditing: context.editable,
    sections: sections.map(([title, ids]) => ({
      title,
      items: ids.map((id) => {
        const command = bindings.get(id === 'fold' && bindings.has('unfold') ? 'unfold' : id) ??
          bindStudioCommand(id, () => {}, { enabled: false });
        return {
          id: command.id,
          label: command.label,
          accelerator: accelerator(command),
          enabled: command.enabled && !composing && commandInContext(command, context),
        };
      }),
    })),
  };
}

type Snapshot = ReturnType<typeof studioMenuPresentation> & { scope: string };
interface NativeMenuHost {
  publish(snapshot: Snapshot | null): Promise<void>;
}

declare global {
  interface Window {
    readonly __FEEDBAX_STUDIO_MENU__?: NativeMenuHost;
  }
}

let publishing = Promise.resolve();

function publish(host: NativeMenuHost, snapshot: Snapshot | null) {
  publishing = publishing.then(() => host.publish(snapshot)).catch((error: unknown) => {
    console.error('Studio menu could not be published', error);
  });
}

export function useStudioNativeMenu(commands: readonly StudioCommand[], paletteOpen: boolean) {
  const current = useRef({ commands, paletteOpen });
  const refresh = useRef<() => void>();

  useLayoutEffect(() => {
    current.current = { commands, paletteOpen };
    refresh.current?.();
  });

  useLayoutEffect(() => {
    const host = window.__FEEDBAX_STUDIO_MENU__;
    if (!host) return;
    let scope = '';
    let lastPresentation = '';
    let composing = false;
    let mounted = true;
    const context = () => studioCommandContext(document.activeElement, current.current.paletteOpen);
    const sync = () => {
      if (!mounted) return;
      const presentation = studioMenuPresentation(current.current.commands, context(), composing);
      const serialized = JSON.stringify(presentation);
      if (serialized === lastPresentation) return;
      lastPresentation = serialized;
      scope = crypto.randomUUID();
      publish(host, { scope, ...presentation });
    };
    function dispatch(event: Event) {
      if (!(event instanceof CustomEvent) || event.detail?.scope !== scope || composing || !document.hasFocus()) return;
      const command = current.current.commands.find(({ id }) => id === event.detail.id && menuIds.has(id));
      if (command && commandInContext(command, context())) command.invoke();
    }
    const focusChanged = () => queueMicrotask(sync);
    const compositionStarted = () => { composing = true; sync(); };
    const compositionEnded = () => { composing = false; sync(); };
    refresh.current = sync;
    window.addEventListener(nativeCommandEvent, dispatch);
    document.addEventListener('focusin', focusChanged);
    document.addEventListener('focusout', focusChanged);
    document.addEventListener('compositionstart', compositionStarted);
    document.addEventListener('compositionend', compositionEnded);
    sync();
    return () => {
      mounted = false;
      refresh.current = undefined;
      window.removeEventListener(nativeCommandEvent, dispatch);
      document.removeEventListener('focusin', focusChanged);
      document.removeEventListener('focusout', focusChanged);
      document.removeEventListener('compositionstart', compositionStarted);
      document.removeEventListener('compositionend', compositionEnded);
      publish(host, null);
    };
  }, []);
}
