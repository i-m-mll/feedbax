// @vitest-environment jsdom

import { act, render } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { useState } from 'react';
import { holdWhile, useCommandSource, useHeldGestures, useWindowCommands } from './heldGestures';
import { bindStudioCommand, type StudioCommand } from './studioCommands';

interface State {
  readonly settled: boolean;
  readonly failed?: boolean;
  /** The commands that apply now. */
  readonly applying: readonly ('undo' | 'redo' | 'run')[];
}

function workspace(log: string[]) {
  let commands: Record<string, StudioCommand> = {};
  function Workspace({ settled, failed = false, applying }: State) {
    useHeldGestures(settled, failed);
    commands = Object.fromEntries((['undo', 'redo', 'run'] as const).map((id) =>
      [id, bindStudioCommand(id, () => log.push(id), { enabled: applying.includes(id) })]));
    return null;
  }
  const view = render(<Workspace settled applying={['undo', 'redo', 'run']} />);
  return {
    update: (state: State) => act(() => view.rerender(<Workspace {...state} />)),
    invoke: (id: string) => act(() => commands[id].invoke()),
  };
}

it('runs the gestures held while unsettled in the order made, each as it applies once settled', () => {
  const log: string[] = [];
  const { update, invoke } = workspace(log);
  update({ settled: false, applying: ['undo'] });
  invoke('undo');
  invoke('redo');
  invoke('run');
  expect(log).toEqual([]);
  update({ settled: true, applying: ['undo', 'redo', 'run'] });
  expect(log).toEqual(['undo', 'redo', 'run']);
});

it('skips a held gesture that no longer applies and still runs the next', () => {
  const log: string[] = [];
  const { update, invoke } = workspace(log);
  update({ settled: false, applying: ['undo', 'redo', 'run'] });
  invoke('undo');
  invoke('redo');
  invoke('run');
  update({ settled: true, applying: ['undo', 'run'] });
  expect(log).toEqual(['undo', 'run']);
});

it('discards what is held when a save fails', () => {
  const log: string[] = [];
  const { update, invoke } = workspace(log);
  update({ settled: false, applying: ['undo', 'redo', 'run'] });
  invoke('undo');
  invoke('run');
  update({ settled: false, failed: true, applying: ['undo', 'redo', 'run'] });
  update({ settled: true, failed: true, applying: ['undo', 'redo', 'run'] });
  expect(log).toEqual([]);
  invoke('run');
  expect(log).toEqual(['run']);
});

it('resolves a held window command to the window\'s command of the state it finds', () => {
  const log: string[] = [];
  let run: StudioCommand | undefined;
  function Workspace({ settled, selected }: { settled: boolean; selected: string }) {
    useHeldGestures(settled, false);
    run = bindStudioCommand('duplicateStep', () => log.push(selected), { subject: selected });
    useWindowCommands([run]);
    return null;
  }
  const view = render(<Workspace settled selected="train" />);
  act(() => view.rerender(<Workspace settled={false} selected="train" />));
  act(() => run!.invoke());
  act(() => view.rerender(<Workspace settled selected="train-2" />));
  expect(log).toEqual(['train-2']);
});

it('runs a view command at once while unsettled, unless a gesture made before it is held', () => {
  const log: string[] = [];
  let commands: StudioCommand[] = [];
  function Workspace({ settled }: { settled: boolean }) {
    useHeldGestures(settled, false);
    commands = [
      bindStudioCommand('undo', () => log.push('undo')),
      bindStudioCommand('model', () => log.push('model')),
    ];
    return null;
  }
  const view = render(<Workspace settled={false} />);
  act(() => commands[1].invoke());
  expect(log).toEqual(['model']);
  act(() => commands[0].invoke());
  act(() => commands[1].invoke());
  expect(log).toEqual(['model']);
  act(() => view.rerender(<Workspace settled />));
  expect(log).toEqual(['model', 'undo', 'model']);
});

it('does nothing for a held gesture on an object no longer mounted, and runs one whose object is', () => {
  const log: string[] = [];
  const stops: Record<string, StudioCommand> = {};
  function Row({ job }: { job: string }) {
    stops[job] = bindStudioCommand('stop', () => log.push(job), { instance: job });
    return null;
  }
  function Workspace({ settled, jobs }: { settled: boolean; jobs: readonly string[] }) {
    useHeldGestures(settled, false);
    return jobs.map((job) => <Row key={job} job={job} />);
  }
  const view = render(<Workspace settled={false} jobs={['a', 'b']} />);
  act(() => stops.a.invoke());
  act(() => stops.b.invoke());
  act(() => view.rerender(<Workspace settled jobs={['b']} />));
  expect(log).toEqual(['b']);
});

it('resolves a held gesture from a closed menu among the commands its mounted source makes', () => {
  const log: string[] = [];
  let rename: StudioCommand | undefined;
  function Menu({ name }: { name: string }) {
    useCommandSource(() => bindStudioCommand('renameStep', () => log.push(name), { subject: 'train' }));
    return null;
  }
  function Workspace({ settled, menu }: { settled: boolean; menu: string | null }) {
    useHeldGestures(settled, false);
    return menu && <Menu name={menu} />;
  }
  const view = render(<Workspace settled={false} menu="first" />);
  // The open menu draws its items, and the gesture is made on one; the menu has closed when the workspace settles.
  act(() => { rename = bindStudioCommand('renameStep', () => log.push('first'), { subject: 'train' }); });
  act(() => rename!.invoke());
  act(() => view.rerender(<Workspace settled menu="second" />));
  expect(log).toEqual(['second']);
  act(() => view.rerender(<Workspace settled={false} menu="second" />));
  act(() => rename!.invoke());
  act(() => view.rerender(<Workspace settled menu={null} />));
  expect(log).toEqual(['second']);
});

it('outside a workspace, holds a gesture made while a request is in flight and runs it once the request is done', async () => {
  vi.useFakeTimers();
  const log: string[] = [];
  let stop: StudioCommand | undefined;
  let finish!: () => void;
  let request!: Promise<void>;
  function Row() {
    const [pending, setPending] = useState(false);
    stop = bindStudioCommand('stop', () => log.push('stop'), { instance: 'job' });
    request ??= holdWhile(async () => {
      await new Promise<void>((resolve) => { finish = resolve; });
      setPending(!pending);
    });
    return null;
  }
  render(<Row />);
  act(() => stop!.invoke());
  expect(log).toEqual([]);
  await act(async () => {
    finish();
    await request;
  });
  await act(() => vi.runAllTimersAsync());
  expect(log).toEqual(['stop']);
  vi.useRealTimers();
});
