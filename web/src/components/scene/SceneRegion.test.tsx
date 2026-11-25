// @vitest-environment jsdom
import userEvent from '@testing-library/user-event';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DrawerSections } from '@/features/projectWindow/DrawerForms';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { elementKey, type ElementAddress } from '@/features/projectWindow/elementSelection';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneLayers, sceneContents } from './SceneRegion';
import { fittedCamera, type Scene, type SceneElement, type SceneEnd, type SceneLayer, type SceneView } from './sceneView';

afterEach(cleanup);
beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
});

const sections: DrawerSections = { isOpen: (_, initially = true) => initially, setOpen: () => {} };

function element(id: string, layer: SceneLayer, shape: SceneElement['shape']): SceneElement {
  const address: ElementAddress = { document: 'arm', address: { composites: ['mechanics'], element: { kind: 'component', id } } };
  return { key: elementKey(address), address, name: id, layer, shape, values: [] };
}

const mechanism: ElementAddress = { document: 'arm', address: { composites: [], element: { kind: 'component', id: 'mechanics' } } };
const scene: Scene = { groups: [{ key: elementKey(mechanism), address: mechanism, name: 'mechanics', elements: [
  element('upper', 'bodies', { kind: 'segment', points: [[0, 0], [0.2, 0.4]] }),
  element('shoulder', 'joints', { kind: 'point', point: [0, 0] }),
  element('flexor', 'muscles', { kind: 'path', points: [[0.05, 0], [0.15, 0.3]] }),
] }], rest: [], tracks: [], ends: [], relations: [], places: [] };

const noLayersHidden = new Set<SceneLayer>();

describe('Scene regions', () => {
  it('shares one selection by address between canvas, Outline and inspector, and Layers hides one class', async () => {
    function View() {
      const [selected, select] = useState<string | null>(null);
      const onSelect = (address: ElementAddress | null) => select(address && elementKey(address));
      const [hidden, setHidden] = useState<ReadonlySet<SceneLayer>>(noLayersHidden);
      const [camera, onCameraChange] = useState(fittedCamera);
      const view: SceneView = { kind: 'drawn', scene };
      return (
        <>
          <SceneLayers layers={sceneContents(view).layers} hidden={hidden} onToggle={(layer) => setHidden(new Set([layer]))} />
          <SceneCanvas view={view} selected={selected} onSelect={onSelect}
            camera={camera} onCameraChange={onCameraChange} hiddenLayers={hidden} />
          <SceneDrawer sections={sections} scene={scene} selected={selected} onSelect={onSelect} />
          <SceneElementInspector scene={scene} selected={selected} />
        </>
      );
    }
    render(<View />);
    const canvas = within(screen.getByRole('group', { name: studioName('scene') }));
    const select = (name: string) => canvas.queryByRole('button', { name: studioAccessibleName('select', name) });
    fireEvent.click(screen.getByRole('treeitem', { name: 'upper' }));
    expect(select('upper')?.getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(select('shoulder')!);
    expect(screen.getByRole('treeitem', { name: 'shoulder' }).getAttribute('aria-selected')).toBe('true');
    expect(screen.getByRole('region', { name: 'shoulder' })).toBeTruthy();
    await userEvent.click(screen.getByRole('button', { name: studioName('layers') }));
    await userEvent.click(await screen.findByRole('menuitemcheckbox', { name: studioName('muscles') }));
    expect(select('flexor')).toBeNull();
    expect(select('upper')).not.toBeNull();
    expect(select('shoulder')?.getAttribute('aria-pressed')).toBe('true');
  });

  it('exposes a failed replay and retries through the input command', () => {
    const retry = vi.fn();
    render(<SceneCanvas view={{ kind: 'failed', issue: { reason: 'Replay unavailable', detail: 'The selected output could not be read.' }, retry }}
      selected={null} onSelect={vi.fn()} camera={fittedCamera}
      onCameraChange={vi.fn()} hiddenLayers={noLayersHidden} />);
    expect(screen.getByRole('alert')).toBeDefined();
    expect(screen.queryByRole('group', { name: studioName('scene') })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: studioName('retry') }));
    expect(retry).toHaveBeenCalledOnce();
  });


  // Catches a Relate that writes the wrong ends: a drag that does not end at the place it is released over, a fixed
  // point placed elsewhere than where it is dropped, a click that relates two places, and an anchor no signal reads
  // taken as an end.
  it('relates the place a drag starts at to the place, or the world point, it ends at', () => {
    const hand: SceneEnd = { key: 'hand', name: 'hand', point: [0, 0],
      place: { kind: 'anchor', address: { composites: [], element: { kind: 'component', id: 'plant' } } } };
    const target: SceneEnd = { key: 'target', name: 'target', point: [0.15, 0.3], place: { kind: 'variable', id: 'target' } };
    const unread: SceneEnd = { ...hand, key: 'unread', name: 'elbow', point: [0.05, 0], reason: 'No signal reads elbow' };
    const relate = vi.fn(async () => undefined);
    vi.spyOn(SVGElement.prototype, 'getBoundingClientRect').mockReturnValue(
      { left: 0, top: 0, width: 860, height: 520, x: 0, y: 0, right: 860, bottom: 520, toJSON: () => ({}) });
    Object.assign(Element.prototype, { setPointerCapture() {}, releasePointerCapture() {}, hasPointerCapture: () => true });
    render(<SceneCanvas view={{ kind: 'drawn', scene: { ...scene, ends: [hand, target, unread] } }} selected={null} onSelect={vi.fn()}
      camera={fittedCamera} onCameraChange={vi.fn()} hiddenLayers={noLayersHidden} tool="relate" onRelate={relate} />);
    const canvas = screen.getByRole('group', { name: studioName('scene') });
    const at = (end: SceneEnd) => {
      const circle = canvas.querySelector(`[data-scene-end="${end.key}"] circle`)!;
      return { clientX: Number(circle.getAttribute('cx')), clientY: Number(circle.getAttribute('cy')) };
    };
    const drag = (from: SceneEnd, to: { clientX: number; clientY: number }) => {
      fireEvent.pointerDown(canvas.querySelector(`[data-scene-end="${from.key}"]`)!, { button: 0, pointerId: 1, ...at(from) });
      fireEvent.pointerMove(canvas, { pointerId: 1, ...to });
      fireEvent.pointerUp(canvas, { pointerId: 1, ...to });
    };
    drag(hand, at(target));
    expect(relate).toHaveBeenLastCalledWith(hand.place, target.place);
    const [, upperEnd] = [...canvas.querySelectorAll('[data-scene-layer="bodies"] circle')];
    drag(target, { clientX: Number(upperEnd.getAttribute('cx')), clientY: Number(upperEnd.getAttribute('cy')) });
    const [, dropped] = relate.mock.lastCall! as unknown as [unknown, { kind: string; position: [number, number] }];
    expect(dropped.kind).toBe('point');
    expect(dropped.position[0]).toBeCloseTo(0.2, 3);
    expect(dropped.position[1]).toBeCloseTo(0.4, 3);
    drag(hand, at(hand));
    expect(relate).toHaveBeenLastCalledWith(hand.place, undefined);
    relate.mockClear();
    drag(unread, at(target));
    drag(hand, at(unread));
    expect(relate).not.toHaveBeenCalled();
  });
});
