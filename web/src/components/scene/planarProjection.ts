import type { ResolvedScene } from '@/features/scenario/projections';

export const WORKSPACE_SVG_WIDTH = 860;
export const WORKSPACE_SVG_HEIGHT = 520;
const WORKSPACE_PADDING = 56;

export type ScenePoint = [number, number];

export interface SceneBounds {
  min: ScenePoint;
  max: ScenePoint;
}

export function includePoint(bounds: SceneBounds, point: ScenePoint) {
  bounds.min = [Math.min(bounds.min[0], point[0]), Math.min(bounds.min[1], point[1])];
  bounds.max = [Math.max(bounds.max[0], point[0]), Math.max(bounds.max[1], point[1])];
}

export function sceneBounds(scene: ResolvedScene): SceneBounds {
  const bounds: SceneBounds = {
    min: [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY],
    max: [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY],
  };

  for (const anchor of scene.anchors) {
    if (anchor.position) includePoint(bounds, anchor.position);
  }
  for (const element of scene.elements) {
    if (element.geometry.kind === 'polyline' || element.geometry.kind === 'points' || element.geometry.kind === 'link') {
      for (const point of element.geometry.points) includePoint(bounds, point);
    } else if (element.geometry.kind === 'bounds') {
      includePoint(bounds, element.geometry.min);
      includePoint(bounds, element.geometry.max);
    }
  }

  if (!Number.isFinite(bounds.min[0]) || !Number.isFinite(bounds.min[1])) {
    return { min: [-1, -1], max: [1, 1] };
  }

  const width = Math.max(bounds.max[0] - bounds.min[0], 0.25);
  const height = Math.max(bounds.max[1] - bounds.min[1], 0.25);
  const pad = Math.max(width, height) * 0.12;
  return {
    min: [bounds.min[0] - pad, bounds.min[1] - pad],
    max: [bounds.max[0] + pad, bounds.max[1] + pad],
  };
}

export function fitScale(bounds: SceneBounds, viewportWidth = WORKSPACE_SVG_WIDTH, viewportHeight = WORKSPACE_SVG_HEIGHT): number {
  const width = Math.max(bounds.max[0] - bounds.min[0], 0.1);
  const height = Math.max(bounds.max[1] - bounds.min[1], 0.1);
  const padding = Math.min(WORKSPACE_PADDING, Math.min(viewportWidth, viewportHeight) / 8);
  return Math.min(
    (viewportWidth - padding * 2) / width,
    (viewportHeight - padding * 2) / height
  );
}

export function niceScaleLength(targetMeters: number): number {
  if (!Number.isFinite(targetMeters) || targetMeters <= 0) return 0.1;
  const exponent = Math.floor(Math.log10(targetMeters));
  const base = 10 ** exponent;
  const normalized = targetMeters / base;
  const step = normalized >= 5 ? 5 : normalized >= 2 ? 2 : 1;
  return step * base;
}
