/** Delays before each successive attempt to reach the service again, the last repeated. */
export const reconnectDelays = [250, 500, 1000, 2000, 4000] as const;

export function reconnectDelay(attempt: number) {
  return reconnectDelays[Math.min(attempt, reconnectDelays.length - 1)];
}
