/**
 * Runs a view's commands one at a time, in the order they were issued. No command is dropped: one issued while another
 * is in flight waits for it and then runs, and its own result (sent, or refused with a reason) is returned to the caller.
 */
export function commandQueue() {
  let tail: Promise<unknown> = Promise.resolve();
  return function run<T>(command: () => Promise<T>): Promise<T> {
    const result = tail.then(command);
    tail = result.catch(() => undefined);
    return result;
  };
}

/**
 * A value that a queued command can wait for. `next` resolves with the value as soon as one is set that the command
 * accepts, which may be at once; the commands behind it in the queue wait with it.
 */
export function awaitedValue<T>() {
  let value: T | undefined;
  let waiting: { accept: (value: T) => boolean; resolve: (value: T) => void }[] = [];
  return {
    set(next: T | undefined) {
      value = next;
      if (next === undefined) return;
      const ready = waiting.filter(({ accept }) => accept(next));
      waiting = waiting.filter((waiter) => !ready.includes(waiter));
      for (const { resolve } of ready) resolve(next);
    },
    next(accept: (value: T) => boolean = () => true): Promise<T> {
      if (value !== undefined && accept(value)) return Promise.resolve(value);
      return new Promise((resolve) => waiting.push({ accept, resolve }));
    },
  };
}
