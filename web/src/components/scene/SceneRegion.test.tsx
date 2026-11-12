// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { normalizeTrainingTrajectoryPayload } from '@/features/scenario/liveTraining';
import type { ResolvedScene } from '@/features/scenario/projections';
import { WORKSPACE_REPLAY_FIXTURE } from '@/features/scenario/workspaceReplay';
import { SceneClockControls } from './SceneClockControls';
import { SceneCanvas, SceneDrawer, SceneElementInspector, SceneHeaderControls } from './SceneRegion';
import { fittedCamera, sceneTimeline, type SceneOverlay, type SceneView } from './sceneView';

afterEach(cleanup);

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
  it('selects a supplied trial independently of the entity selection', () => {
    const chooseTrial = vi.fn();
    const chooseEntity = vi.fn();
    render(<SceneDrawer scene={scene} selectedId="arm" onSelect={chooseEntity} trials={{
      choices: [{ id: 'trial-a', label: 'Straight reach' }, { id: 'trial-b', label: 'Curl reach' }],
      selectedId: 'trial-a',
      onSelect: chooseTrial,
    }} />);
    fireEvent.change(screen.getByRole('combobox', { name: 'Trials' }), { target: { value: 'trial-b' } });
    expect(chooseTrial).toHaveBeenCalledWith('trial-b');
    expect(chooseEntity).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Arm' }).getAttribute('aria-pressed')).toBe('true');
  });

  it('shares entity selection between separately mounted canvas, drawer and inspector', () => {
    function View() {
      const [selectedId, onSelect] = useState<string | null>(null);
      const [hiddenOverlays, setHidden] = useState<ReadonlySet<SceneOverlay>>(noOverlaysHidden);
      const [camera, onCameraChange] = useState(fittedCamera);
      const view: SceneView = { kind: 'geometry', scene };
      return (
        <>
          <SceneHeaderControls view={view} camera={camera} onCameraChange={onCameraChange}
            hiddenOverlays={hiddenOverlays} onToggleOverlay={(overlay) => setHidden(new Set([overlay]))} />
          <SceneCanvas view={view} time={0} selectedId={selectedId} onSelect={onSelect}
            camera={camera} onCameraChange={onCameraChange} hiddenOverlays={hiddenOverlays} />
          <SceneDrawer scene={scene} selectedId={selectedId} onSelect={onSelect} />
          <SceneElementInspector scene={scene} selectedId={selectedId} />
        </>
      );
    }
    render(<View />);
    const canvas = within(screen.getByRole('group', { name: studioName('scene') }));
    fireEvent.click(screen.getByRole('button', { name: 'Arm' }));
    expect(canvas.getByRole('button', { name: studioAccessibleName('select', 'Arm') }).getAttribute('aria-pressed')).toBe('true');
    fireEvent.click(canvas.getByRole('button', { name: studioAccessibleName('select', 'Far goal') }));
    expect(screen.getByRole('button', { name: 'Far goal' }).getAttribute('aria-pressed')).toBe('true');
    expect(within(screen.getByRole('region', { name: studioName('details') })).getByRole('heading').textContent).toBe('Far goal');
    fireEvent.click(screen.getByRole('button', { name: studioName('objectives') }));
    expect(canvas.queryByRole('button', { name: studioAccessibleName('select', 'Reaching cost') })).toBeNull();
    expect(canvas.getByRole('button', { name: studioAccessibleName('select', 'Far goal') }).getAttribute('aria-pressed')).toBe('true');
  });

  it('scrubs a replay and a separate view on the same clock, with no clock for geometry', () => {
    const replay = WORKSPACE_REPLAY_FIXTURE.trials![0];
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
    const { container, rerender } = render(<View view={{ kind: 'replay', scene, trial: replay }} />);
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
    const frame = normalizeTrainingTrajectoryPayload({ effector: [[-0.5, 0], [0.6, 0]] });
    const latest = normalizeTrainingTrajectoryPayload({ effector: [[0.6, 0], [-0.5, 0]] });
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
