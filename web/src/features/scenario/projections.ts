import type { StudioSelectorRef } from '@/generated/studioContracts';

type SceneEntityKind =
  | 'graph_node'
  | 'graph_port'
  | 'graph_edge'
  | 'task_object'
  | 'task_data'
  | 'task_binding'
  | 'mechanics_object'
  | 'objective_term'
  | 'retained_observable'
  | 'probe'
  | 'metric'
  | 'temporal_event'
  | 'temporal_phase'
  | 'stage_protocol'
  | 'artifact_overlay';

export type SceneValidationSeverity = 'warning' | 'error';

export interface ResolvedSceneValidationMessage {
  type: string;
  severity: SceneValidationSeverity;
  message: string;
  entity_id?: string | null;
  path?: string | null;
}

export interface ResolvedSceneAnchor {
  id: string;
  entity_id: string;
  local_id: string;
  label: string;
  semantic_role: string;
  position: [number, number] | null;
  selectable: boolean;
  hoverable: boolean;
  interaction_roles: string[];
  objective_roles: string[];
  frame: string;
  selector: StudioSelectorRef | null;
  metadata: Record<string, unknown>;
}

export type ResolvedSceneGeometry =
  | { kind: 'polyline'; points: Array<[number, number]> }
  | { kind: 'bounds'; min: [number, number]; max: [number, number] }
  | { kind: 'points'; points: Array<[number, number]> }
  | { kind: 'link'; points: Array<[number, number]> }
  | { kind: 'none' };

export interface ResolvedSceneElement {
  id: string;
  entity_id: string;
  local_id: string;
  archetype: string;
  anchor_ids: string[];
  frame: string;
  scale_invariant: boolean;
  style: Record<string, unknown>;
  geometry: ResolvedSceneGeometry;
  metadata: Record<string, unknown>;
}

export interface ResolvedSceneEntity {
  id: string;
  label: string;
  kind: SceneEntityKind;
  summary: string | null;
  related_entity_ids: string[];
  anchor_ids: string[];
  element_ids: string[];
}

export interface ResolvedScene {
  scenario_id: string | null;
  frame: string;
  units: string;
  entities: ResolvedSceneEntity[];
  anchors: ResolvedSceneAnchor[];
  elements: ResolvedSceneElement[];
  required_selectors: StudioSelectorRef[];
  validation: ResolvedSceneValidationMessage[];
}
