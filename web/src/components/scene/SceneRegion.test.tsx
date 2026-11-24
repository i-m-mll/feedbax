// @vitest-environment jsdom
import userEvent from '@testing-library/user-event';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DrawerSections } from '@/features/projectWindow/DrawerForms';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { elementKey, type ElementAddress } from '@/features/projectWindow/elementSelection';
import { SceneClockControls } from './SceneClockControls';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneLayers, sceneContents } from './SceneRegion';
import { emptyScene, fittedCamera, sceneTimeline, type Scene, type SceneElement, type SceneLayer, type SceneView } from './sceneView';

afterEach(cleanup);
beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  });
  Element.prototype.scrollIntoView = vi.fn();
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
] }] };

const noLayersHidden = new Set<SceneLayer>();

describe('Scene regions', () => {
  it('shares one selection by address between canvas, Outline and inspector, and Layers hides one class', async () => {
    function View() {
      const [selected, select] = useState<string | null>(null);
      const onSelect = (address: ElementAddress | null) => select(address && elementKey(address));
      const [hidden, setHidden] = useState<ReadonlySet<SceneLayer>>(noLayersHidden);
      const [camera, onCameraChange] = useState(fittedCamera);
      const view: SceneView = { kind: 'geometry', scene };
      return (
        <>
          <SceneLayers layers={sceneContents(view).layers} hidden={hidden} onToggle={(layer) => setHidden(new Set([layer]))} />
          <SceneCanvas view={view} time={0} selected={selected} onSelect={onSelect}
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

  it('scrubs a replay and a separate view on the same clock, with no clock for geometry', () => {
    const position = 'arm/effector';
    const trial = {
      id: 'trial-a', inits: {}, inputs: {}, present: {}, variables: {}, variable_defined: {}, timeline: [],
      states: { [position]: { samples: [[0, 0], [0.1, 0.1], [0.2, 0.2]], units: 'm', frame: 'world' } },
    };
    const onSelect = vi.fn();
    function View({ view }: { view: SceneView }) {
      const [time, onTimeChange] = useState(0);
      return (
        <>
          <SceneCanvas view={view} time={time} selected={null} onSelect={onSelect}
            camera={fittedCamera} onCameraChange={vi.fn()} hiddenLayers={noLayersHidden} />
          <div data-testid="results-cursor">{time}</div>
          <SceneClockControls timeline={sceneTimeline(view)} time={time} speed={1}
            onTimeChange={onTimeChange} onSpeedChange={vi.fn()} />
        </>
      );
    }
    const { container, rerender } = render(<View view={{ kind: 'replay', scene: emptyScene, trial, position, stateTime: [0, 0.6, 1.2] }} />);
    const initialX = container.querySelector('[data-scene-cursor]')?.getAttribute('cx');
    expect(initialX).toBeDefined();
    fireEvent.change(screen.getByRole('slider'), { target: { value: '0.60' } });
    expect(screen.getByTestId('results-cursor').textContent).toBe('0.6');
    expect(container.querySelector('[data-scene-cursor]')?.getAttribute('cx')).not.toBe(initialX);
    rerender(<View view={{ kind: 'geometry', scene }} />);
    expect(screen.queryByRole('slider')).toBeNull();
    expect(container.querySelector('[data-scene-cursor]')).toBeNull();
  });

  it('exposes a failed replay and retries through the input command', () => {
    const retry = vi.fn();
    render(<SceneCanvas view={{ kind: 'failed', issue: { reason: 'Replay unavailable', detail: 'The selected output could not be read.' }, retry }}
      time={0} selected={null} onSelect={vi.fn()} camera={fittedCamera}
      onCameraChange={vi.fn()} hiddenLayers={noLayersHidden} />);
    expect(screen.getByRole('alert')).toBeDefined();
    expect(screen.queryByRole('group', { name: studioName('scene') })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: studioName('retry') }));
    expect(retry).toHaveBeenCalledOnce();
  });

});
