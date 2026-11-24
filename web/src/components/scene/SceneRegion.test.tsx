// @vitest-environment jsdom
import userEvent from '@testing-library/user-event';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DrawerSections } from '@/features/projectWindow/DrawerForms';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { normalizeTrainingTrajectoryPayload } from '@/features/scenario/liveTraining';
import type { ResolvedScene } from '@/features/scenario/projections';
import { SceneClockControls } from './SceneClockControls';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneLayers, sceneLayers } from './SceneRegion';
import { fittedCamera, sceneTimeline, type SceneOverlay, type SceneView } from './sceneView';

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

const scene: ResolvedScene = {
  scenario_id: null,
  frame: 'world.xy',
  units: 'm',
  required_selectors: [],
  validation: [],
  entities: [
    { id: 'arm', label: 'Arm', kind: 'mechanics_object', summary: null, anchor_ids: [], element_ids: ['arm-line'], related_entity_ids: [] },
    { id: 'goal', label: 'Far goal', kind: 'task_object', summary: null, anchor_ids: [], element_ids: ['goal-point'], related_entity_ids: [] },
    { id: 'penalty', label: 'Reaching cost', kind: 'objective_term', summary: null, anchor_ids: [], element_ids: ['penalty-line'], related_entity_ids: [] },
  ],
  anchors: [],
  elements: [
    { id: 'arm-line', entity_id: 'arm', local_id: 'line', archetype: 'planar_chain', frame: 'world.xy', anchor_ids: [],
      scale_invariant: false, style: {}, metadata: {}, geometry: { kind: 'polyline', points: [[0, 0], [0.2, 0.4]] } },
    { id: 'goal-point', entity_id: 'goal', local_id: 'goal', archetype: 'target', frame: 'world.xy', anchor_ids: [],
      scale_invariant: true, style: { glyph: 'target' }, metadata: {}, geometry: { kind: 'points', points: [[0.5, 0.4]] } },
    { id: 'penalty-line', entity_id: 'penalty', local_id: 'penalty', archetype: 'objective_link', frame: 'world.xy', anchor_ids: [],
      scale_invariant: true, style: {}, metadata: {}, geometry: { kind: 'link', points: [[0.2, 0.4], [0.5, 0.4]] } },
  ],
};

const noOverlaysHidden = new Set<SceneOverlay>();

describe('Scene regions', () => {
  it('shares entity selection between separately mounted canvas, drawer and inspector', async () => {
    function View() {
      const [selectedId, onSelect] = useState<string | null>(null);
      const [hiddenOverlays, setHidden] = useState<ReadonlySet<SceneOverlay>>(noOverlaysHidden);
      const [camera, onCameraChange] = useState(fittedCamera);
      const view: SceneView = { kind: 'geometry', scene };
      return (
        <>
          <SceneLayers overlays={sceneLayers(view).overlays}
            hiddenOverlays={hiddenOverlays} onToggleOverlay={(overlay) => setHidden(new Set([overlay]))} />
          <SceneCanvas view={view} time={0} selectedId={selectedId} onSelect={onSelect}
            camera={camera} onCameraChange={onCameraChange} hiddenOverlays={hiddenOverlays} />
          <SceneDrawer sections={sections} scene={scene} selectedId={selectedId} onSelect={onSelect} />
          <SceneElementInspector scene={scene} selectedId={selectedId} />
        </>
      );
    }
    render(<View />);
    const canvas = within(screen.getByRole('group', { name: studioName('scene') }));
    fireEvent.click(screen.getByRole('treeitem', { name: 'Arm' }));
    expect(canvas.getByRole('button', { name: studioAccessibleName('select', 'Arm') }).getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(canvas.getByRole('button', { name: studioAccessibleName('select', 'Far goal') }));
    expect(screen.getByRole('treeitem', { name: 'Far goal' }).getAttribute('aria-selected')).toBe('true');
    expect(screen.getByRole('region', { name: 'Far goal' })).toBeTruthy();
    await userEvent.click(screen.getByRole('button', { name: studioName('layers') }));
    await userEvent.click(await screen.findByRole('menuitemcheckbox', { name: studioName('objectives') }));
    expect(canvas.queryByRole('button', { name: studioAccessibleName('select', 'Reaching cost') })).toBeNull();
    expect(canvas.getByRole('button', { name: studioAccessibleName('select', 'Far goal') }).getAttribute('aria-pressed')).toBe('true');
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
          <SceneCanvas view={view} time={time} selectedId={null} onSelect={onSelect}
            camera={fittedCamera} onCameraChange={vi.fn()} hiddenOverlays={noOverlaysHidden} />
          <div data-testid="results-cursor">{time}</div>
          <SceneClockControls timeline={sceneTimeline(view)} time={time} speed={1}
            onTimeChange={onTimeChange} onSpeedChange={vi.fn()} />
        </>
      );
    }
    const { container, rerender } = render(<View view={{ kind: 'replay', scene, trial, position, stateTime: [0, 0.6, 1.2] }} />);
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
      time={0} selectedId={null} onSelect={vi.fn()} camera={fittedCamera}
      onCameraChange={vi.fn()} hiddenOverlays={noOverlaysHidden} />);
    expect(screen.getByRole('alert')).toBeDefined();
    expect(screen.queryByRole('group', { name: studioName('scene') })).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: studioName('retry') }));
    expect(retry).toHaveBeenCalledOnce();
  });

  it('updates the live marker from the incoming frame without a playback clock', () => {
    const frameFor = (samples: number[][]) => normalizeTrainingTrajectoryPayload({
      tracks: {
        effector: {
          anchor_id: 'effector',
          selector: { namespace: 'graph_output', compact: 'graph_output:effector', role: 'observed' },
          samples, dim: 2, dtype: 'float32',
        },
      },
    });
    const frame = frameFor([[-0.5, 0], [0.6, 0]]);
    const latest = frameFor([[0.6, 0], [-0.5, 0]]);
    const drawing = {
      time: 0, selectedId: null, onSelect: vi.fn(), camera: fittedCamera,
      onCameraChange: vi.fn(), hiddenOverlays: noOverlaysHidden,
    };
    const view = (frame: typeof latest) => <SceneCanvas {...drawing} view={{ kind: 'live', scene, frame }} />;
    const { container, rerender } = render(view(frame));
    const firstX = Number(container.querySelector('[data-scene-cursor]')?.getAttribute('cx'));
    rerender(view(latest));
    const lastX = Number(container.querySelector('[data-scene-cursor]')?.getAttribute('cx'));
    expect(lastX).toBeLessThan(firstX);
    expect(sceneTimeline({ kind: 'live', scene, frame: latest })).toBeNull();
  });
});
