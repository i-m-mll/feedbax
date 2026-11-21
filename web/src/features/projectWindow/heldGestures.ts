import { useEffect, useLayoutEffect, useReducer } from 'react';

/** What a held gesture resolves to when it runs: whether it applies then, and what it does. */
export interface HeldTarget {
  readonly enabled: boolean;
  perform(): void;
}

/**
 * Gestures made while the workspace is unsettled, in the order made. Each is resolved to the command of the state when
 * it runs: it runs if it applies then and does nothing if it does not.
 */
export class HeldGestures<Gesture> {
  private gestures: Gesture[] = [];

  get size() {
    return this.gestures.length;
  }

  hold(gesture: Gesture) {
    this.gestures.push(gesture);
  }

  /** Resolves and runs the first held gesture; false when none is held. */
  runNext(resolve: (gesture: Gesture) => HeldTarget | undefined) {
    if (!this.gestures.length) return false;
    const target = resolve(this.gestures.shift()!);
    if (target?.enabled) target.perform();
    return true;
  }

  discard() {
    this.gestures = [];
  }
}

/**
 * A held gesture. One made on a window command (by key, the palette or the window's menu) resolves to the window's
 * command of that id when it runs, acting on the selection it finds; one made on an object's own command (its button
 * or context menu) resolves to the latest binding of that object's command.
 */
interface Gesture {
  readonly id: string;
  readonly key: string;
  readonly window: boolean;
}

interface Binding {
  readonly key: string;
  readonly target: HeldTarget;
  /** Whether the command only changes how the window shows the workspace, so need not wait for it to settle. */
  readonly view: boolean;
}

const bindings = new WeakMap<object, Binding>();

const gate = {
  /** Whether a workspace is mounted and holds gestures. */
  active: false,
  settled: true,
  held: new HeldGestures<Gesture>(),
  /** The latest binding of each object command's key. */
  bound: new Map<string, HeldTarget>(),
  /** The window's commands as last rendered. */
  window: [] as readonly { readonly id: string }[],
  rerender: undefined as (() => void) | undefined,
};

/** Records what `command` does, and makes it the latest binding of `key` unless the key is empty. */
export function bindGesture(command: object, key: string, target: HeldTarget, view = false) {
  bindings.set(command, { key, target, view });
  if (key) gate.bound.set(key, target);
}

/**
 * Invokes a command: at once when nothing is held before it and the workspace is settled or the command only changes
 * the view, else held.
 *
 * A command run marks the workspace unsettled until the workspace renders again, so a gesture made before that
 * render waits for the state the command leaves.
 */
export function invokeGesture(command: { readonly id: string }) {
  const { key, target, view } = bindings.get(command)!;
  if (!gate.active) {
    if (target.enabled) target.perform();
    return;
  }
  if (gate.held.size || (!gate.settled && !view)) {
    gate.held.hold({ id: command.id, key, window: gate.window.includes(command) });
    return;
  }
  if (target.enabled) run(target);
}

function run(target: HeldTarget) {
  gate.settled = false;
  target.perform();
  gate.rerender?.();
}

function resolve({ id, key, window }: Gesture): HeldTarget | undefined {
  if (!window) return gate.bound.get(key);
  const command = gate.window.find((item) => item.id === id);
  return command && bindings.get(command)?.target;
}

/** Makes `commands` the window's commands, which a gesture held on one of them resolves among. */
export function useWindowCommands(commands: readonly { readonly id: string }[]) {
  useLayoutEffect(() => {
    gate.window = commands;
  });
}

/**
 * Holds the gestures made while the workspace is unsettled, and runs them one per render once it settles.
 * A failed save discards what is held.
 */
export function useHeldGestures(settled: boolean, failed: boolean) {
  const [, rerender] = useReducer((count: number) => count + 1, 0);
  useLayoutEffect(() => {
    gate.active = true;
    gate.rerender = rerender;
    return () => {
      gate.active = false;
      gate.settled = true;
      gate.rerender = undefined;
      gate.held.discard();
      gate.bound.clear();
      gate.window = [];
    };
  }, []);
  useLayoutEffect(() => {
    gate.settled = settled;
  });
  useEffect(() => {
    if (failed) gate.held.discard();
  }, [failed]);
  useEffect(() => {
    if (!settled || !gate.held.size) return;
    gate.held.runNext((gesture) => {
      const target = resolve(gesture);
      return target && { enabled: target.enabled, perform: () => run(target) };
    });
    rerender();
  });
}
