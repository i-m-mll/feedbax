import { describe, expect, it } from 'vitest';
import { ANALYSIS_CANVAS_INTERACTION_PROPS } from './AnalysisCanvas';

describe('AnalysisCanvas interaction policy', () => {
  it('disables default keyboard deletion and volatile node dragging', () => {
    expect(ANALYSIS_CANVAS_INTERACTION_PROPS).toEqual({
      deleteKeyCode: null,
      nodesDraggable: false,
    });
  });
});
