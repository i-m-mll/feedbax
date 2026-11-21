import { useEffect, useLayoutEffect, useReducer, useRef } from 'react';

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
 * or context menu) resolves to that object's command as bound while the gesture runs, and does nothing when no
 * mounted object binds it.
 */
interface Gesture {
  readonly id: string;
  readonly key: string;
  readonly window: boolean;
  /** The bindings made before the gesture. */
  readonly at: number;
  /** What each registered gesture context read when the gesture was made. */
  readonly context: readonly unknown[];
}

/** Module state a command reads from the gesture that invokes it, such as the control and the input that made it. */
const contexts: { capture(): unknown; restore(value: unknown): void }[] = [];

/**
 * Registers state a command reads from the event that invoked it. A held gesture keeps what `capture` read when it was
 * made, and `restore` puts it back just before the gesture runs.
 */
export function gestureContext<T>(capture: () => T, restore: (value: T) => void) {
  contexts.push({ capture, restore: restore as (value: unknown) => void });
}

interface Binding {
  readonly key: string;
  readonly target: HeldTarget;
  /** Whether the command only changes how the window shows the workspace, so need not wait for it to settle. */
  readonly view: boolean;
}

const bindings = new WeakMap<object, Binding>();

/** How many bindings have been made; a binding's count orders it among them. */
let bindCount = 0;

/** The command factories of mounted menus whose items are not drawn while closed. */
const sources = new Set<{ current: (() => unknown) | undefined }>();

/**
 * Registers the factory of commands a component draws only on demand, such as a closed menu's. A gesture held on one of
 * them resolves among what the factory makes while the component is mounted.
 */
export function useCommandSource(factory: (() => unknown) | undefined) {
  const source = useRef(factory);
  useLayoutEffect(() => {
    source.current = factory;
  });
  useLayoutEffect(() => {
    sources.add(source);
    return () => {
      sources.delete(source);
    };
  }, []);
}

const gate = {
  /** Whether a workspace is mounted; it settles and runs what is held. */
  active: false,
  settled: true,
  /** Work started with `holdWhile` and not yet done; gestures wait for it as for the workspace. */
  working: 0,
  held: new HeldGestures<Gesture>(),
  /** The latest binding of each object command's key, with its count. */
  bound: new Map<string, { readonly target: HeldTarget; readonly at: number }>(),
  /** The bindings made before the workspace's latest render; the mounted objects' bindings follow them. */
  rendered: 0,
  /** The bindings made when the workspace last committed. */
  committed: 0,
  /** The window's commands as last rendered. */
  window: [] as readonly { readonly id: string }[],
  rerender: undefined as (() => void) | undefined,
};

/** Records what `command` does, and makes it the latest binding of `key` unless the key is empty. */
export function bindGesture(command: object, key: string, target: HeldTarget, view = false) {
  bindings.set(command, { key, target, view });
  bindCount += 1;
  if (key) gate.bound.set(key, { target, at: bindCount });
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
  if (gate.held.size || ((!gate.settled || gate.working > 0) && !view)) {
    gate.held.hold({
      id: command.id, key, window: gate.window.includes(command), at: bindCount,
      context: contexts.map(({ capture }) => capture()),
    });
    return;
  }
  if (target.enabled) run(target);
}

function run(target: HeldTarget) {
  if (gate.active) gate.settled = false;
  target.perform();
  gate.rerender?.();
}

/** Runs the next held gesture, resolved to the command of the state now. */
function runNext() {
  return gate.held.runNext((gesture) => {
    const target = resolve(gesture);
    return target && {
      enabled: target.enabled,
      perform: () => {
        contexts.forEach(({ restore }, index) => restore(gesture.context[index]));
        run(target);
      },
    };
  });
}

/**
 * Runs `work`, holding gestures made until it is done as the workspace holds those made while it settles. Outside a
 * workspace, what is held runs once the work is done and its result has rendered.
 */
export async function holdWhile<T>(work: () => Promise<T>): Promise<T> {
  gate.working += 1;
  try {
    return await work();
  } finally {
    gate.working -= 1;
    if (gate.active) gate.rerender?.();
    else setTimeout(drainOutside);
  }
}

/** Outside a workspace, runs one held gesture per task, so each finds the state the one before it rendered. */
function drainOutside() {
  if (gate.active || gate.working || !runNext()) return;
  setTimeout(drainOutside);
}

/**
 * The command a held gesture runs now. An object's command must have been bound since the workspace last rendered,
 * which every mounted object does by rendering or through its command source; outside a workspace, since the gesture.
 */
function resolve({ id, key, window, at }: Gesture): HeldTarget | undefined {
  if (!window) {
    const since = gate.active ? Math.max(at, gate.rendered) : at;
    const mounted = () => {
      const found = gate.bound.get(key);
      return found && found.at > since ? found.target : undefined;
    };
    return mounted() ?? (sources.forEach(({ current }) => current?.()), mounted());
  }
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
    gate.rendered = gate.committed;
    gate.committed = bindCount;
  });
  useEffect(() => {
    if (failed) gate.held.discard();
  }, [failed]);
  useEffect(() => {
    if (!settled || gate.working || !gate.held.size) return;
    runNext();
    rerender();
  });
}
