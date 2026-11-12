// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { fitProjectWindow } from './projectWindowLayout';
import {
  defaultProjectWindowState, projectWindowSlotView, showProjectWindowView, useProjectWindowState,
} from './projectWindowState';

afterEach(() => window.localStorage.clear());

describe('Studio window arrangement', () => {
  it('folds a slot under space pressure while preserving its preferred width and the drawer', () => {
    const state = showProjectWindowView({ ...defaultProjectWindowState, secondaryWidth: 390 }, 'scene', 'secondary');
    const wide = fitProjectWindow(1230, 700, state);
    const narrow = fitProjectWindow(850, 700, state);
    expect(wide.secondary.open).toBe(true);
    expect(narrow.secondary.open).toBe(false);
    expect(narrow.secondary.size).toBeGreaterThan(0);
    expect(narrow.detailsDrawer.size).toBe(wide.detailsDrawer.size);
    expect(fitProjectWindow(1230, 700, state).secondary.size).toBe(390);
    expect(state.focusedSlot).toBe('secondary');
  });

  it('retains view preferences when the selection temporarily has different views', () => {
    const state = showProjectWindowView({ ...defaultProjectWindowState, mode: 'map' }, 'scene', 'secondary');
    expect(state.mode).toBe('both');
    expect(projectWindowSlotView(state, 'secondary', ['runs', 'analysis'])).toBe('runs');
    expect(projectWindowSlotView(state, 'secondary', ['runs', 'scene'])).toBe('scene');
  });

  it('restores independent app-local preferences and ignores non-arrangement fields', () => {
    const first = renderHook(() => useProjectWindowState('one'));
    act(() => first.result.current[1]((state) => ({
      ...showProjectWindowView(state, 'scene', 'secondary'),
      mode: 'details',
      mapHeight: 305,
      mapDrawerOpen: false,
      detailsDrawerWidth: 193,
      extraProjectData: 'outside the arrangement',
    })));
    first.unmount();

    const reopened = renderHook(({ id }) => useProjectWindowState(id), { initialProps: { id: 'one' } });
    expect(reopened.result.current[0]).toMatchObject({
      mode: 'details',
      secondaryView: 'scene',
      mapHeight: 305,
      mapDrawerOpen: false,
      detailsDrawerWidth: 193,
    });
    expect(reopened.result.current[0]).not.toHaveProperty('extraProjectData');
    reopened.rerender({ id: 'two' });
    expect(reopened.result.current[0]).toEqual(defaultProjectWindowState);
  });
});
