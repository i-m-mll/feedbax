import type { StudioRegionIssue } from '@/components/activity/StudioRegion';
import type { ElementAddress } from '@/features/projectWindow/elementSelection';
import type { ScenePoint } from './planarProjection';

/** A class of element the Scene draws, in Layers' order; Layers shows or hides each. */
export const sceneLayers = ['bodies', 'joints', 'muscles', 'variables', 'signals'] as const;
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
  /** What the element is in the shown trial; absent when it has no place there. */
  readonly shape?: SceneShape;
  /** The places the element may take, one per trial that may be shown, drawn hollow. */
  readonly possible?: readonly ScenePoint[];
  readonly values: readonly SceneValue[];
  /** Which end of the trial a task element marks: the start the Model is placed at, or a goal. */
  readonly mark?: 'start' | 'goal';
  /** Why the element is not drawn as authored, marked on it. */
  readonly problem?: string;
}

/** A mechanism, or the task, with the elements of it the Scene draws. */
export interface SceneGroup {
  readonly key: string;
  readonly address: ElementAddress;
  readonly name: string;
  readonly elements: readonly SceneElement[];
  readonly problem?: string;
}

export interface Scene {
  readonly groups: readonly SceneGroup[];
  /** The mechanisms at rest, drawn faint beneath a recorded pose. */
  readonly rest: readonly SceneShape[];
  /** Each recorded world-frame track, drawn as a path, the picked trial's last. */
  readonly tracks: readonly SceneTrack[];
}

/**
 * A recorded world-frame track: its whole path, how many of its points are drawn (all of them, or those up to the
 * playhead under Trails), where it is at the playhead, and whether it is the picked trial's, drawn heavier.
 */
export interface SceneTrack {
  readonly points: readonly ScenePoint[];
  readonly drawn: number;
  readonly head?: ScenePoint;
  readonly picked: boolean;
}

export const emptyScene: Scene = { groups: [], rest: [], tracks: [] };

export type SceneView =
  | { kind: 'drawn'; scene: Scene }
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
