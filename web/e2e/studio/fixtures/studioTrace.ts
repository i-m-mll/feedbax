import { readFile } from 'node:fs/promises';

type TraceEvent = {
  name: string;
  cat?: string;
  ph: string;
  pid: number;
  tid: number;
  ts: number;
  args?: {
    layerTreeId?: number;
    hasPartialUpdate?: boolean;
    frameSeqId?: number;
    data?: { type?: string; duration?: number; interactionId?: number };
  };
};

const frameWindows = ['drag-100-0', 'drag-100-1', 'drag-100-2', 'drag-100-3', 'scroll', 'playing'];

export async function summarizeStudioTrace(path: string) {
  const { traceEvents } = JSON.parse(await readFile(path, 'utf8')) as { traceEvents: TraceEvent[] };

  function marker(name: string) {
    const matches = traceEvents.filter(event => event.name === name && event.cat?.includes('blink.user_timing'));
    if (matches.length !== 1) throw new Error(`Expected one Chromium timing mark for ${name}, found ${matches.length}`);
    return matches[0];
  }

  function eventsIn(name: string) {
    const start = marker(`studio-${name}-start`);
    const end = marker(`studio-${name}-end`);
    if (start.pid !== end.pid || start.tid !== end.tid || end.ts <= start.ts) {
      throw new Error(`Invalid Chromium timing window for ${name}`);
    }
    return {
      start,
      events: traceEvents.filter(event => event.pid === start.pid && event.ts >= start.ts && event.ts <= end.ts),
    };
  }

  const typing = eventsIn('typing');
  const keydowns = typing.events.filter(event =>
    event.tid === typing.start.tid && event.name === 'EventTiming' &&
    event.ph === 'b' && event.args?.data?.type === 'keydown',
  );
  const keydownMs = keydowns.map(event => {
    const duration = event.args?.data?.duration;
    if (duration === undefined || !Number.isFinite(duration) || duration < 0) {
      throw new Error('Chromium EventTiming keydown is missing a valid duration');
    }
    return duration;
  });

  const frames = frameWindows.map(name => {
    const { start, events } = eventsIn(name);
    const drawn = events.filter(event => event.name === 'DrawFrame' && event.args?.layerTreeId !== undefined);
    const streams = new Set(drawn.map(event => `${event.tid}:${event.args?.layerTreeId}`));
    if (streams.size !== 1) throw new Error(`Expected one renderer frame stream for ${name}, found ${streams.size}`);
    const stream = events.filter(event =>
      event.tid === drawn[0].tid && event.args?.layerTreeId === drawn[0].args?.layerTreeId,
    );
    return {
      name,
      beginFrames: stream.filter(event => event.name === 'BeginFrame').length,
      drawFrames: drawn.length,
      rendererDrops: stream.filter(event => event.name === 'DroppedFrame').map(event => ({
        offsetMs: (event.ts - start.ts) / 1000,
        sequence: event.args?.frameSeqId,
        partialUpdate: event.args?.hasPartialUpdate,
      })),
    };
  });

  return { keydownMs, frames };
}
