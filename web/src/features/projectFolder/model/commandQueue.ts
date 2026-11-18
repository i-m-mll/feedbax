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
