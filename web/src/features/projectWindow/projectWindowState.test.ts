// @vitest-environment jsdom
import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { fitProjectWindow } from './projectWindowLayout';
import {
  defaultProjectWindowState, projectWindowSlotView, showProjectWindowView, useProjectWindowState,
} from './projectWindowState';

afterEach(() => vi.unstubAllGlobals());

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

  it('restores independent native preferences without persisting focus or authored fields', async () => {
    const preferences = new Map<string, unknown>();
    vi.stubGlobal('__FEEDBAX_STUDIO_MENU__', {});
    const fetch = vi.fn(async (path: string, init?: RequestInit) => {
      if (init?.method === 'PUT') {
        preferences.set(path, JSON.parse(String(init.body)));
        return new Response(null, { status: 204 });
      }
      return Response.json(preferences.get(path) ?? null);
    });
    vi.stubGlobal('fetch', fetch);
    const first = renderHook(() => useProjectWindowState('one'));
    act(() => first.result.current[1]((state) => ({
      ...showProjectWindowView(state, 'scene', 'secondary'),
      mode: 'details',
      mapHeight: 305,
      mapDrawerOpen: false,
      detailsDrawerWidth: 193,
      extraProjectData: 'outside the arrangement',
    })));
    await waitFor(() => expect(preferences.size).toBe(1));
    first.unmount();

    const reopened = renderHook(({ id }) => useProjectWindowState(id), { initialProps: { id: 'one' } });
    await waitFor(() => expect(reopened.result.current[0]).toMatchObject({
      mode: 'details',
      secondaryView: 'scene',
      focusedSlot: 'primary',
      mapHeight: 305,
      mapDrawerOpen: false,
      detailsDrawerWidth: 193,
    }));
    expect(preferences.values().next().value).not.toHaveProperty('focusedSlot');
    expect(preferences.values().next().value).not.toHaveProperty('extraProjectData');
    await act(async () => reopened.rerender({ id: 'two' }));
    expect(reopened.result.current[0]).toEqual(defaultProjectWindowState);
  });

  it('keeps ordinary browser arrangements in memory', () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    const first = renderHook(() => useProjectWindowState('one'));
    act(() => first.result.current[1]((state) => ({ ...state, mapHeight: 400 })));
    expect(first.result.current[0].mapHeight).toBe(400);
    first.unmount();
    expect(renderHook(() => useProjectWindowState('one')).result.current[0].mapHeight)
      .toBe(defaultProjectWindowState.mapHeight);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('acknowledges the latest arrangement when close follows an in-flight write', async () => {
    vi.stubGlobal('__FEEDBAX_STUDIO_MENU__', {});
    let finishFirst: () => void = () => {};
    const first = new Promise<Response>((resolve) => {
      finishFirst = () => resolve(new Response(null, { status: 204 }));
    });
    const writes: number[] = [];
    vi.stubGlobal('fetch', vi.fn(async (_path: string, init?: RequestInit) => {
      if (init?.method !== 'PUT') return Response.json(null);
      writes.push(JSON.parse(String(init.body)).mapHeight);
      return writes.length === 1 ? first : new Response(null, { status: 204 });
    }));
    const windowState = renderHook(() => useProjectWindowState('one'));
    act(() => windowState.result.current[1]((state) => ({ ...state, mapHeight: 400 })));
    await waitFor(() => expect(writes).toEqual([400]));
    act(() => windowState.result.current[1]((state) => ({ ...state, mapHeight: 500 })));
    await act(async () => {
      const flush = window.__FEEDBAX_FLUSH_WINDOW_PREFERENCES__!();
      finishFirst();
      await flush;
    });
    expect(writes).toEqual([400, 500]);
    act(() => windowState.result.current[1]((state) => ({ ...state, mapHeight: 600 })));
    expect(windowState.result.current[0].mapHeight).toBe(500);
    await window.__FEEDBAX_FLUSH_WINDOW_PREFERENCES__!(false);
    act(() => windowState.result.current[1]((state) => ({ ...state, mapHeight: 600 })));
    expect(windowState.result.current[0].mapHeight).toBe(600);
  });

  it('retries a failed preference save when the native session reconnects', async () => {
    vi.stubGlobal('__FEEDBAX_STUDIO_MENU__', {});
    const report = vi.spyOn(console, 'error').mockImplementation(() => {});
    const writes: number[] = [];
    vi.stubGlobal('fetch', vi.fn(async (_path: string, init?: RequestInit) => {
      if (init?.method !== 'PUT') return Response.json(null);
      writes.push(JSON.parse(String(init.body)).mapHeight);
      return new Response(null, { status: writes.length === 1 ? 503 : 204 });
    }));
    const windowState = renderHook(() => useProjectWindowState('one'));
    act(() => windowState.result.current[1]((state) => ({ ...state, mapHeight: 460 })));
    await waitFor(() => expect(report).toHaveBeenCalledOnce());
    await act(async () => { window.dispatchEvent(new Event('studio-native-reconnected')); });
    expect(writes).toEqual([460, 460]);
    report.mockRestore();
  });
});
