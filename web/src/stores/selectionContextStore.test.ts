import { beforeEach, describe, expect, it } from 'vitest';
import { selectedIdSet, useSelectionContextStore } from '@/stores/selectionContextStore';

beforeEach(() => {
  useSelectionContextStore.getState().reset();
});

describe('useSelectionContextStore', () => {
  it('keeps selected ids and focused id as distinct selection context fields', () => {
    useSelectionContextStore.getState().setContext({
      stage: 'stage:train',
      collection: 'collection:training-runs',
      selectedIds: ['run-a', 'run-b', 'run-a'],
      focusedId: 'run-b',
    });

    const { context } = useSelectionContextStore.getState();
    expect(context).toMatchObject({
      stage: 'stage:train',
      collection: 'collection:training-runs',
      selectedIds: ['run-a', 'run-b'],
      focusedId: 'run-b',
    });
    expect(selectedIdSet(context)).toEqual(new Set(['run-a', 'run-b']));
  });

  it('supports linked hover preview while preserving click-committed focus', () => {
    const store = useSelectionContextStore.getState();
    store.focusId('run-a');
    store.previewFocus('run-b');

    expect(useSelectionContextStore.getState().context.focusedId).toBe('run-a');
    expect(useSelectionContextStore.getState().previewId).toBe('run-b');

    useSelectionContextStore.getState().setSyncMode('decoupled');
    useSelectionContextStore.getState().previewFocus('run-c');

    expect(useSelectionContextStore.getState().previewId).toBeNull();
    expect(useSelectionContextStore.getState().context.focusedId).toBe('run-a');
  });

  it('supports lineage projection hover and click-through focus targets', () => {
    const store = useSelectionContextStore.getState();
    store.setContext({
      stage: 'stage:train',
      collection: 'collection:training-runs',
      selectedIds: ['train-a'],
      focusedId: 'train-a',
    });
    store.previewFocus('eval-a');

    expect(useSelectionContextStore.getState().previewId).toBe('eval-a');
    expect(useSelectionContextStore.getState().context.focusedId).toBe('train-a');

    store.setContext({
      stage: 'stage:eval',
      collection: 'collection:evaluation-runs',
      selectedIds: ['eval-a'],
      focusedId: 'eval-a',
    });

    expect(useSelectionContextStore.getState().context).toMatchObject({
      stage: 'stage:eval',
      collection: 'collection:evaluation-runs',
      selectedIds: ['eval-a'],
      focusedId: 'eval-a',
    });

    store.setSyncMode('decoupled');
    store.previewFocus('analysis-a');

    expect(useSelectionContextStore.getState().previewId).toBeNull();
    expect(useSelectionContextStore.getState().context.focusedId).toBe('eval-a');
  });

  it('prunes context when a collection changes without carrying stale selected ids', () => {
    const store = useSelectionContextStore.getState();
    store.setContext({
      stage: 'stage:train',
      collection: 'collection:training-runs',
      selectedIds: ['run-a', 'run-b'],
      focusedId: 'run-b',
    });
    store.syncCollection('stage:train', 'collection:training-runs', ['run-a']);

    expect(useSelectionContextStore.getState().context).toMatchObject({
      selectedIds: ['run-a'],
      focusedId: null,
    });

    store.syncCollection('stage:eval', 'collection:selected-training-runs', ['run-a']);
    expect(useSelectionContextStore.getState().context).toMatchObject({
      stage: 'stage:eval',
      collection: 'collection:selected-training-runs',
      selectedIds: [],
      focusedId: null,
    });
  });
});
