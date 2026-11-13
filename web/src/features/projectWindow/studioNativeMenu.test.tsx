// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { ProjectWindow } from './ProjectWindow';
import { defaultProjectWindowState } from './projectWindowState';
import { bindStudioCommand, type StudioCommand } from './studioCommands';

type Snapshot = NonNullable<Parameters<NonNullable<Window['__FEEDBAX_STUDIO_MENU__']>['publish']>[0]>;

function studio(
  run: () => void, projectName = 'First', withScene = false, rename?: () => void,
  commands: readonly StudioCommand[] = [
    bindStudioCommand('run', run), bindStudioCommand('stop', () => {}, { enabled: false }),
  ],
) {
  return <ProjectWindow state={defaultProjectWindowState} onStateChange={() => {}}
    projectName={projectName} availableViews={withScene ? ['runs', 'scene'] : ['runs']}
    renderView={() => ({ body: <input aria-label="Training field" /> })}
    map={<button>Map item</button>} mapDrawer={null} compactMap={<button>Compact item</button>}
    activity={<div data-studio-activity-panel><button>Activity item</button></div>}
    commands={[
      ...commands,
      ...(rename ? [bindStudioCommand('rename', rename)] : []),
    ]} />;
}

function setup() {
  vi.stubGlobal('ResizeObserver', class { observe() {} disconnect() {} });
  vi.spyOn(document, 'hasFocus').mockReturnValue(true);
}

function menu() {
  let latest: Snapshot | null = null;
  const publish = vi.fn(async (snapshot: Snapshot | null) => { latest = snapshot; });
  vi.stubGlobal('__FEEDBAX_STUDIO_MENU__', { publish });
  return {
    latest: () => latest,
    select: (id: string, scope: string) => {
      act(() => { window.dispatchEvent(new CustomEvent('studio-native-command', { detail: { scope, id } })); });
    },
  };
}

function item(snapshot: Snapshot | null, id: string) {
  return snapshot?.sections.flatMap(({ items }) => items).find((item) => item.id === id);
}

afterEach(async () => {
  cleanup();
  await Promise.resolve();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it.each([
  { id: 'undo', key: 'z', code: 'KeyZ', accelerator: 'Cmd+Z' },
  { id: 'redo', key: 'Z', code: 'KeyZ', shiftKey: true, accelerator: 'Shift+Cmd+Z' },
  { id: 'run', key: 'r', code: 'KeyR', accelerator: 'Cmd+R' },
  { id: 'stop', key: '.', code: 'Period', accelerator: 'Cmd+.' },
] as const)('leaves $accelerator to the native menu and dispatches its command once', async ({ id, accelerator, ...key }) => {
  setup();
  const bridge = menu();
  const invoke = vi.fn();
  render(studio(vi.fn(), 'First', false, undefined, [bindStudioCommand(id, invoke)]));
  await waitFor(() => expect(item(bridge.latest(), id)).toMatchObject({ accelerator, enabled: true }));

  expect(fireEvent.keyDown(document.body, { ...key, metaKey: true })).toBe(true);
  expect(invoke).not.toHaveBeenCalled();
  bridge.select(id, bridge.latest()!.scope);
  expect(invoke).toHaveBeenCalledTimes(1);
});

it('refuses native commands from another project scope', async () => {
  setup();
  const bridge = menu();
  const first = vi.fn();
  const second = vi.fn();
  const view = render(studio(first));
  await waitFor(() => expect(item(bridge.latest(), 'run')).toMatchObject({
    id: 'run', label: 'Run', accelerator: 'Cmd+R', enabled: true,
  }));
  expect(item(bridge.latest(), 'stop')).toMatchObject({ accelerator: 'Cmd+.', enabled: false });
  expect(item(bridge.latest(), 'scene')).toMatchObject({ accelerator: 'Cmd+6', enabled: false });
  const scope = bridge.latest()!.scope;

  expect(fireEvent.keyDown(document.body, { key: 'r', code: 'KeyR', metaKey: true })).toBe(true);
  expect(first).not.toHaveBeenCalled();
  bridge.select('run', scope);
  expect(first).toHaveBeenCalledTimes(1);
  bridge.select('unknown', scope);
  expect(first).toHaveBeenCalledTimes(1);

  view.rerender(<div>{studio(second, 'Second', true)}</div>);
  await waitFor(() => expect(item(bridge.latest(), 'scene')).toMatchObject({ enabled: true }));
  bridge.select('run', scope);
  expect(second).not.toHaveBeenCalled();
  bridge.select('run', bridge.latest()!.scope);
  expect(second).toHaveBeenCalledTimes(1);
  vi.mocked(document.hasFocus).mockReturnValue(false);
  bridge.select('run', bridge.latest()!.scope);
  expect(second).toHaveBeenCalledTimes(1);
});

it('updates menu availability and rechecks the focused control on delivery', async () => {
  setup();
  const bridge = menu();
  const run = vi.fn();
  const undo = vi.fn();
  render(studio(run, 'First', false, undefined, [bindStudioCommand('run', run), bindStudioCommand('undo', undo)]));
  await waitFor(() => expect(item(bridge.latest(), 'run')?.enabled).toBe(true));
  const scope = bridge.latest()!.scope;

  act(() => { screen.getByRole('textbox', { name: 'Training field' }).focus(); });
  expect(fireEvent.keyDown(document.activeElement!, { key: 'z', code: 'KeyZ', metaKey: true })).toBe(true);
  bridge.select('undo', scope);
  expect(undo).not.toHaveBeenCalled();
  bridge.select('run', scope);
  expect(run).not.toHaveBeenCalled();
  await waitFor(() => expect(bridge.latest()?.textEditing).toBe(true));
  expect(item(bridge.latest(), 'run')?.enabled).toBe(false);

  act(() => { screen.getByRole('button', { name: 'Activity item' }).focus(); });
  await waitFor(() => expect(bridge.latest()?.textEditing).toBe(false));
  expect(item(bridge.latest(), 'run')?.enabled).toBe(false);
  expect(item(bridge.latest(), 'showMap')).toMatchObject({ accelerator: 'Ctrl+Cmd+1', enabled: true });
  bridge.select('run', bridge.latest()!.scope);
  expect(run).not.toHaveBeenCalled();
});

it('keeps the web shortcut in a browser without the native host', () => {
  setup();
  const run = vi.fn();
  render(studio(run));
  expect(fireEvent.keyDown(document.body, { key: 'r', code: 'KeyR', metaKey: true })).toBe(false);
  expect(run).toHaveBeenCalledTimes(1);
});

it('routes File Rename to the focused project', async () => {
  setup();
  const bridge = menu();
  const rename = vi.fn();
  const run = vi.fn();
  const view = render(studio(run, 'First', false, rename));
  await waitFor(() => expect(bridge.latest()?.sections.find(({ title }) => title === 'File')?.items)
    .toEqual([{ id: 'rename', label: 'Rename', accelerator: undefined, enabled: true }]));
  bridge.select('rename', bridge.latest()!.scope);
  expect(rename).toHaveBeenCalledTimes(1);
  view.rerender(studio(run));
  await waitFor(() => expect(item(bridge.latest(), 'rename')?.enabled).toBe(false));
  bridge.select('rename', bridge.latest()!.scope);
  expect(rename).toHaveBeenCalledTimes(1);
});
