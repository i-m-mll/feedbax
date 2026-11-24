import { useLayoutEffect, useState, type RefObject } from 'react';
import { useStore } from 'zustand';
import { createStore, type StoreApi } from 'zustand/vanilla';

/** A horizontal extent on screen, in viewport pixels. */
export interface AxisSpan {
  readonly left: number;
  readonly width: number;
}

/** The scrubber's track, which the shelf's time axis spans, so that a time is at one x in both. */
export type TimeAxisStore = StoreApi<{ span?: AxisSpan }>;

export function useTimeAxisStore(): TimeAxisStore {
  const [store] = useState(() => createStore<{ span?: AxisSpan }>(() => ({})));
  return store;
}

/** A shelf drawn without a scrubber, which lays out its own axis. */
const unmeasured: TimeAxisStore = createStore<{ span?: AxisSpan }>(() => ({}));

/** Calls `measure` whenever `ref`'s element changes size or the window does. */
function useMeasured(ref: RefObject<HTMLElement | null>, measure: (element: HTMLElement) => void) {
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => measure(element);
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    window.addEventListener('resize', update);
    return () => {
      observer.disconnect();
      window.removeEventListener('resize', update);
    };
  });
}

/** Publishes the extent of `ref`'s element, the scrubber's track, to `store`. */
export function usePublishedAxis(store: TimeAxisStore, ref: RefObject<HTMLElement | null>) {
  useMeasured(ref, (element) => {
    const { left, width } = element.getBoundingClientRect();
    const span = store.getState().span;
    if (span?.left !== left || span.width !== width) store.setState({ span: { left, width } });
  });
}

/** The scrubber's track as an extent of `ref`'s element, the shelf: its offset from the element's left edge and its width. */
export function useAxisWithin(store: TimeAxisStore | undefined, ref: RefObject<HTMLElement | null>): AxisSpan | undefined {
  const span = useStore(store ?? unmeasured, (state) => state.span);
  const [origin, setOrigin] = useState<number>();
  useMeasured(ref, (element) => {
    const left = element.getBoundingClientRect().left;
    if (left !== origin) setOrigin(left);
  });
  return store && span && origin !== undefined ? { left: span.left - origin, width: span.width } : undefined;
}
