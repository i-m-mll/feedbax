import { useEffect, useRef, useState, useSyncExternalStore, type ReactNode, type RefObject } from 'react';
import { createPortal, flushSync } from 'react-dom';
import { followPress, type Point } from './pointerPress';
import { usePopupLayer } from './popupLayer';

/** A kind of thing a drag carries from one place to another, typed by the value it carries. */
export interface CarryKind<T> {
  readonly name: string;
  /** Never set: it ties the kind to its value's type. */
  readonly value?: T;
}

export function carryKind<T>(name: string): CarryKind<T> {
  return { name };
}

interface Target {
  kind: CarryKind<unknown> | undefined;
  element: RefObject<HTMLElement | null>;
  takes(value: unknown, at: Point): boolean;
  drop(value: unknown, at: Point): void;
}

/** How a place receives a drag: `takes` says whether it would at a point (by default it would anywhere), `drop` receives it. */
export interface CarryReceiver<T> {
  takes?(value: T, at: Point): boolean;
  drop(value: T, at: Point): void;
}

/** A drag in progress: what it carries, the pointer's position, and the place under the pointer that takes it. */
export interface Carry<T> {
  kind: CarryKind<T>;
  value: T;
  at: Point;
  target: Target | undefined;
}

let carrying: Carry<unknown> | undefined;
const listeners = new Set<() => void>();
const targets = new Set<Target>();

function publish(next: Carry<unknown> | undefined) {
  carrying = next;
  for (const listener of listeners) listener();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

/** The innermost place under the pointer at `at` that takes `value`. */
function targetAt(kind: CarryKind<unknown>, value: unknown, at: Point) {
  const under = document.elementFromPoint(at.x, at.y);
  let found: Target | undefined;
  for (const target of targets) {
    const element = target.element.current;
    if (target.kind !== kind || !element || !under || !element.contains(under) || !target.takes(value, at)) continue;
    if (!found || found.element.current!.contains(element)) found = target;
  }
  return found;
}

/**
 * Carries `value` from the press `event` begins, once the pointer moves: each place that takes `kind` draws it while
 * the pointer is over it, and the one it is released over receives it.
 */
export function carry<T>(event: { pointerId: number; button: number; clientX: number; clientY: number }, kind: CarryKind<T>, value: T) {
  followPress(event, () => {
    const follow = (at: Point) => publish({ kind, value, at, target: targetAt(kind as CarryKind<unknown>, value, at) });
    return {
      move: follow,
      release(at) {
        const target = targetAt(kind as CarryKind<unknown>, value, at);
        // What the drop makes is drawn in the same frame its preview is taken away.
        flushSync(() => {
          target?.drop(value, at);
          publish(undefined);
        });
      },
      cancel: () => publish(undefined),
    };
  });
}

/**
 * Makes `element` a place that receives drags of `kind`. Returns the drag while the pointer is over a point of it that
 * takes the drag, for it to draw what the drop would make.
 */
export function useCarryTarget<T>(element: RefObject<HTMLElement | null>, kind: CarryKind<T> | undefined,
  receiver: CarryReceiver<T>): Carry<T> | undefined {
  const received = useRef(receiver);
  received.current = receiver;
  const [target] = useState<Target>(() => ({
    kind: undefined, element,
    takes: (value, at) => received.current.takes?.(value as T, at) ?? true,
    drop: (value, at) => received.current.drop(value as T, at),
  }));
  useEffect(() => {
    if (!kind) return;
    target.kind = kind as CarryKind<unknown>;
    targets.add(target);
    return () => { targets.delete(target); };
  }, [kind, target]);
  return useSyncExternalStore(subscribe, () => carrying?.target === target ? carrying as Carry<T> : undefined);
}

/** What a drag of `kind` carries, drawn at the pointer while no place under it takes it. */
export function CarryChip<T>({ kind, children }: { kind: CarryKind<T> | undefined; children: (value: T) => ReactNode }) {
  const layer = usePopupLayer();
  const carried = useSyncExternalStore(subscribe, () => kind && carrying?.kind === kind && !carrying.target ? carrying : undefined);
  if (!carried) return null;
  return createPortal(<div className="studio-carry-chip" style={{ left: carried.at.x, top: carried.at.y }}>
    {children(carried.value as T)}
  </div>, layer ?? document.body);
}
