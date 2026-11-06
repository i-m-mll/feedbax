import { describe, expect, it, vi } from 'vitest';
import { consumeAnalysisDeleteKey } from './AnalysisCanvas';

describe('AnalysisCanvas interaction policy', () => {
  it.each(['Delete', 'Backspace'])('consumes analysis-owned %s keys', (key) => {
    const event = {
      key,
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as Parameters<typeof consumeAnalysisDeleteKey>[0];

    consumeAnalysisDeleteKey(event);

    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(event.stopPropagation).toHaveBeenCalledOnce();
  });
});
