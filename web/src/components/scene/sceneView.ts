import type { StudioRegionIssue } from '@/components/activity/StudioRegion';
import type { ElementAddress } from '@/features/projectWindow/elementSelection';
import type { TrialRecord } from '@/generated/studioContracts';
import type { ScenePoint } from './planarProjection';

/** A class of element the Scene draws, in Layers' order; Layers shows or hides each. */
export const sceneLayers = ['bodies', 'joints', 'muscles'] as const;
export type SceneLayer = typeof sceneLayers[number];

export type SceneShape =
  | { kind: 'segment'; points: readonly [ScenePoint, ScenePoint] }
  | { kind: 'point'; point: ScenePoint }
  | { kind: 'path'; points: readonly ScenePoint[] };

/** A position the Inspector shows for an element: in the world frame, or authored in the frame it names. */
export interface SceneValue {
  readonly label: string;
  readonly point: ScenePoint;
}

/** An element the Scene draws, identified by its address in the document that holds it. */
export interface SceneElement {
  /** `elementKey(address)`. */
  readonly key: string;
  readonly address: ElementAddress;
  readonly name: string;
  readonly layer: SceneLayer;
  /** Absent when the element cannot be placed. */
  readonly shape?: SceneShape;
  readonly values: readonly SceneValue[];
  /** Why the element is not drawn as authored, marked on it. */
  readonly problem?: string;
}

/** A mechanism with the elements of it the Scene draws. */
export interface SceneGroup {
  readonly key: string;
  readonly address: ElementAddress;
  readonly name: string;
  readonly elements: readonly SceneElement[];
  readonly problem?: string;
}

export interface Scene {
  readonly groups: readonly SceneGroup[];
}

export const emptyScene: Scene = { groups: [] };

export type SceneDrawing =
  | { kind: 'geometry'; scene: Scene }
  | { kind: 'replay'; scene: Scene; trial: TrialRecord; stateTime: number[]; position: string | null };

export type SceneView =
  | SceneDrawing
  | { kind: 'loading' }
  | { kind: 'empty' }
  | { kind: 'failed'; issue: StudioRegionIssue; retry: () => void };

export function sceneElements(scene: Scene) {
  return scene.groups.flatMap((group) => group.elements);
}

export interface SceneCamera {
  zoom: number;
  pan: { x: number; y: number };
}

export const fittedCamera: SceneCamera = { zoom: 1, pan: { x: 0, y: 0 } };

export function zoomScene(camera: SceneCamera, factor: number): SceneCamera {
  return { ...camera, zoom: Math.max(0.35, Math.min(8, camera.zoom * factor)) };
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
