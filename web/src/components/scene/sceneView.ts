import type { LiveTrainingFrame } from '@/features/scenario/liveTraining';
import type { ResolvedScene, ResolvedSceneEntity } from '@/features/scenario/projections';
import type { TrialRecord } from '@/generated/studioContracts';
import type { StudioRegionIssue } from '@/components/activity/StudioRegion';

export type SceneDrawing =
  | { kind: 'geometry'; scene: ResolvedScene }
  | { kind: 'live'; scene: ResolvedScene; frame: LiveTrainingFrame | null }
  | { kind: 'replay'; scene: ResolvedScene; trial: TrialRecord; stateTime: number[]; position: string | null; sceneAt?: (time: number) => ResolvedScene };

export type SceneView =
  | SceneDrawing
  | { kind: 'loading' }
  | { kind: 'empty' }
  | { kind: 'failed'; issue: StudioRegionIssue; retry: () => void };

export type SceneOverlay = 'objectives' | 'observables' | 'overlays';

export interface SceneCamera {
  zoom: number;
  pan: { x: number; y: number };
}

export const fittedCamera: SceneCamera = { zoom: 1, pan: { x: 0, y: 0 } };

export function zoomScene(camera: SceneCamera, factor: number): SceneCamera {
  return { ...camera, zoom: Math.max(0.35, Math.min(8, camera.zoom * factor)) };
}

export function sceneGeometry(view: SceneDrawing, time: number): ResolvedScene {
  return view.kind === 'replay' && view.sceneAt ? view.sceneAt(time) : view.scene;
}

export function sceneEntityOverlay(entity: ResolvedSceneEntity): SceneOverlay | undefined {
  if (entity.kind === 'objective_term') return 'objectives';
  if (entity.kind === 'retained_observable' || entity.kind === 'probe') return 'observables';
  if (entity.kind === 'artifact_overlay') return 'overlays';
}

export function sceneTimeline(view: SceneView) {
  if (view.kind !== 'replay') return null;
  const { trial, stateTime: frameTimes } = view;
  const duration = frameTimes[frameTimes.length - 1];
  if (frameTimes.length < 2 || !(duration > 0)) return null;
  return {
    id: `trial:${trial.id}`,
    frameTimes,
    duration,
    units: 's',
  };
}

export type SceneTimeline = NonNullable<ReturnType<typeof sceneTimeline>>;
