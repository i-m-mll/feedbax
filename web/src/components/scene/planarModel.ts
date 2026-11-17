import type {
  ResolvedScene, ResolvedSceneAnchor, ResolvedSceneElement, ResolvedSceneValidationMessage,
} from '@/features/scenario/projections';
import type { ScenePlanarModel, StudioPlanarSceneResponse } from '@/generated/studioContracts';
import type { ScenePoint } from './planarProjection';

export type PlanarScenePose =
  | { kind: 'reference' }
  | { kind: 'inspection'; qpos: readonly number[] }
  | { kind: 'recorded'; sample: { modelIdentity: string; qpos: readonly number[] } | null };

type PlanarGeometry = Omit<ScenePlanarModel, 'model_identity'>;

interface PlanarFrame {
  position: ScenePoint;
  angle: number;
}

function pointInWorld(frame: PlanarFrame, [x, y]: readonly number[]): ScenePoint {
  const cosine = Math.cos(frame.angle);
  const sine = Math.sin(frame.angle);
  return [frame.position[0] + x * cosine - y * sine, frame.position[1] + x * sine + y * cosine];
}

function poseGeometry(model: PlanarGeometry, qpos: readonly number[]) {
  if (model.links.length !== model.reference_qpos.length || model.links.some((link, index) => link.qpos_index !== index)) {
    return { ok: false, code: 'scene.invalid_layout', message: 'The model has an invalid planar joint layout.' } as const;
  }
  if (qpos.length !== model.reference_qpos.length || !Array.from(qpos).every(Number.isFinite)) {
    return { ok: false, code: 'scene.invalid_pose', message: 'The pose requires one finite configuration coordinate per joint.' } as const;
  }

  let parent: PlanarFrame = { position: [0, 0], angle: 0 };
  const frames = new Map([[model.world_frame, parent]]);
  const joints = new Map<string, ScenePoint>();
  for (const link of model.links) {
    const pivot = pointInWorld(parent, link.parent_pivot);
    const angle = parent.angle + link.reference_angle + (qpos[link.qpos_index] - model.reference_qpos[link.qpos_index]);
    // The hinge can be at the distal end, so rotate about its local pivot.
    const position = pointInWorld({ position: pivot, angle }, [-link.body_pivot[0], -link.body_pivot[1]]);
    const body = { position, angle };
    const distal = pointInWorld(body, [link.length, 0]);
    if (![...pivot, ...position, ...distal, angle].every(Number.isFinite)) {
      return { ok: false, code: 'scene.invalid_pose', message: 'The pose produces nonfinite planar geometry.' } as const;
    }
    frames.set(`${link.body_id}.proximal`, body);
    frames.set(`${link.body_id}.distal`, { position: distal, angle });
    joints.set(link.joint_id, pivot);
    parent = body;
  }
  return { ok: true, frames, joints } as const;
}

function posePlanarModel(model: ScenePlanarModel, pose: PlanarScenePose) {
  if (pose.kind === 'recorded') {
    if (!pose.sample) return { ok: false, code: 'scene.pose_unavailable', message: 'The recording has no planar pose at this time.' } as const;
    if (pose.sample.modelIdentity !== model.model_identity) {
      return { ok: false, code: 'scene.pose_model_mismatch', message: 'The recorded pose belongs to a different model.' } as const;
    }
    return poseGeometry(model, pose.sample.qpos);
  }
  return posePlanarPreview(model, pose);
}

function posePlanarPreview(model: PlanarGeometry, pose: Exclude<PlanarScenePose, { kind: 'recorded' }>) {
  return poseGeometry(model, pose.kind === 'reference' ? model.reference_qpos : pose.qpos);
}

export function planarSceneEntityId(source: StudioPlanarSceneResponse, nodeId: string) {
  return JSON.stringify([source.project_id, source.document_id, source.node_path, nodeId]);
}

export function resolveProjectPlanarScene(source: StudioPlanarSceneResponse, pose: PlanarScenePose): ResolvedScene {
  const scene: ResolvedScene = {
    scenario_id: null, frame: 'world.xy', units: 'm', entities: [], anchors: [], elements: [], required_selectors: [],
    validation: source.diagnostics.map((diagnostic): ResolvedSceneValidationMessage => ({
      type: diagnostic.code, severity: diagnostic.severity === 'warning' ? 'warning' : 'error', message: diagnostic.message,
      entity_id: diagnostic.node_ids?.length ? planarSceneEntityId(source, diagnostic.node_ids[0]) : null,
    })),
  };
  if (!source.model) return scene;
  const model = source.model;
  const posed = pose.kind !== 'recorded' ? posePlanarPreview(model, pose) : posePlanarModel(source.model, pose);
  if (!posed.ok) {
    scene.validation.push({
      type: posed.code, message: posed.message, severity: 'error',
      entity_id: planarSceneEntityId(source, source.node_path[source.node_path.length - 1]),
    });
    return scene;
  }

  function addAnchor(entityId: string, id: string, label: string, role: string, position: ScenePoint, metadata: Record<string, unknown> = {}) {
    const anchor: ResolvedSceneAnchor = {
      id, entity_id: entityId, local_id: label, label, semantic_role: role, position, selectable: true, hoverable: true,
      interaction_roles: ['selectable', 'hoverable'], objective_roles: [], frame: scene.frame, selector: null, metadata,
    };
    scene.anchors.push(anchor);
    return anchor;
  }

  function addEntity(id: string, label: string, archetype: string, anchors: ResolvedSceneAnchor[], points: ScenePoint[] | null) {
    const element: ResolvedSceneElement = {
      id, entity_id: id, local_id: label, archetype, anchor_ids: anchors.map((anchor) => anchor.id), frame: scene.frame,
      scale_invariant: false, style: {}, geometry: points ? { kind: 'polyline', points } : { kind: 'none' }, metadata: {},
    };
    scene.elements.push(element);
    scene.entities.push({
      id, label, kind: 'mechanics_object', summary: null, related_entity_ids: [],
      anchor_ids: element.anchor_ids, element_ids: [element.id],
    });
  }

  for (const link of model.links) {
    const id = planarSceneEntityId(source, link.body_id);
    const proximal = `${link.body_id}.proximal`;
    const distal = `${link.body_id}.distal`;
    const start = posed.frames.get(proximal)!;
    const end = posed.frames.get(distal)!;
    addEntity(id, link.body_id, 'planar_chain', [
      addAnchor(id, `${id}/proximal`, proximal, 'body', start.position, { angle: start.angle }),
      addAnchor(id, `${id}/distal`, distal, 'endpoint', end.position, { angle: end.angle }),
      addAnchor(id, planarSceneEntityId(source, link.joint_id), link.joint_id, 'joint', posed.joints.get(link.joint_id)!, {
        parent_frame: link.parent_frame, child_frame: link.child_frame, qpos_index: link.qpos_index,
      }),
    ], [start.position, end.position]);
  }

  for (const muscle of model.muscles) {
    const id = planarSceneEntityId(source, muscle.node_id);
    const anchors: ResolvedSceneAnchor[] = [];
    const points: ScenePoint[] = [];
    for (const point of muscle.points ?? []) {
      const frame = posed.frames.get(point.frame);
      if (!frame) {
        scene.validation.push({
          type: 'scene.invalid_frame', severity: 'error', entity_id: id,
          message: `${muscle.node_id} refers to unknown frame ${point.frame}.`,
        });
        break;
      }
      points.push(pointInWorld(frame, point.position));
    }
    if (muscle.points && points.length === muscle.points.length) {
      for (const [index, point] of muscle.points.entries()) {
        const role = index === 0 ? 'origin' : index === muscle.points.length - 1 ? 'insertion' : 'guide';
        anchors.push(addAnchor(id, `${id}/point/${index}`, `${point.frame} ${role}`, role, points[index], {
          authored_frame: point.frame, offset: point.position, muscle: muscle.node_id, point_index: index,
        }));
      }
    }
    addEntity(id, muscle.node_id, 'muscle_path', anchors, anchors.length ? points : null);
  }
  return scene;
}
