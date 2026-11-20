// Neither the node nor the jsdom test environment provides ResizeObserver; components that measure layout need one.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

// jsdom lays nothing out, so a range has no box; components that measure content need one.
if (typeof Range !== 'undefined' && !Range.prototype.getBoundingClientRect) {
  Range.prototype.getBoundingClientRect = () => new DOMRect();
}

// Node defines its own `localStorage`, undefined unless started with a storage file, and it hides jsdom's.
if (typeof window !== 'undefined' && globalThis.localStorage === undefined && 'jsdom' in window) {
  Object.defineProperty(globalThis, 'localStorage', { value: (window as unknown as { jsdom: { window: Window } }).jsdom.window.localStorage, configurable: true });
}
