import { describe, expect, it, vi } from 'vitest';
import {
  ANALYSIS_CANVAS_INTERACTION_PROPS,
  ANALYSIS_INTERACTION_OWNER,
  consumeAnalysisDeleteKey,
} from './AnalysisCanvas';

describe('AnalysisCanvas interaction policy', () => {
  it('disables default keyboard deletion and volatile node dragging', () => {
    expect(ANALYSIS_CANVAS_INTERACTION_PROPS).toEqual({
      deleteKeyCode: null,
      nodesDraggable: false,
    });
  });

  it.each(['Delete', 'Backspace'])('consumes analysis-owned %s keys', (key) => {
    const event = {
      key,
      preventDefault: vi.fn(),
      stopPropagation: vi.fn(),
    } as unknown as Parameters<typeof consumeAnalysisDeleteKey>[0];

    consumeAnalysisDeleteKey(event);

    expect(ANALYSIS_INTERACTION_OWNER).toBe('analysis');
    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(event.stopPropagation).toHaveBeenCalledOnce();
  });
});
