import { elementKey, type ElementAddress } from '@/features/projectWindow/elementSelection';
import type { ScenePlanarModel, StudioPlanarSceneResponse } from '@/generated/studioContracts';
import type { ScenePoint } from './planarProjection';
import type { Scene, SceneElement, SceneLayer, SceneShape, SceneValue } from './sceneView';

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

/**
 * The Scene of a planar interior in a pose: each body, joint and muscle under its Model address, and each diagnostic
 * marked on the element it names, or else on the mechanism.
 */
export function resolveProjectPlanarScene(source: StudioPlanarSceneResponse, pose: PlanarScenePose): Scene {
  const document = source.document_id;
  const mechanismId = source.node_path[source.node_path.length - 1];
  const address = (composites: readonly string[], id: string): ElementAddress =>
    ({ document, address: { composites: [...composites], element: { kind: 'component', id } } });
  const elements: SceneElement[] = [];
  const problems: string[] = [];
  const posed = source.model && (pose.kind !== 'recorded' ? posePlanarPreview(source.model, pose) : posePlanarModel(source.model, pose));
  const add = (id: string, layer: SceneLayer, shape: SceneShape | undefined, values: SceneValue[], problem?: string) => {
    const located = address(source.node_path, id);
    const diagnosed = source.diagnostics.find((item) => item.node_ids?.[0] === id)?.message;
    elements.push({ key: elementKey(located), address: located, name: id, layer, shape, values, problem: problem || diagnosed });
  };

  if (posed && !posed.ok) problems.push(posed.message);
  if (posed?.ok) {
    for (const link of source.model!.links) {
      const start = posed.frames.get(`${link.body_id}.proximal`)!;
      const end = posed.frames.get(`${link.body_id}.distal`)!;
      add(link.body_id, 'bodies', { kind: 'segment', points: [start.position, end.position] }, [
        { label: `${link.body_id}.proximal`, point: start.position },
        { label: `${link.body_id}.distal`, point: end.position },
      ]);
      const pivot = posed.joints.get(link.joint_id)!;
      add(link.joint_id, 'joints', { kind: 'point', point: pivot }, [{ label: link.joint_id, point: pivot }]);
    }
    for (const muscle of source.model!.muscles) {
      const authored = muscle.points ?? [];
      const missing = authored.find((point) => !posed.frames.has(point.frame));
      const values = authored.map((point, index) => ({
        label: `${point.frame} ${index === 0 ? 'origin' : index === authored.length - 1 ? 'insertion' : 'guide'}`,
        point: [point.position[0], point.position[1]] as ScenePoint,
      }));
      add(muscle.node_id, 'muscles', missing || !authored.length ? undefined : {
        kind: 'path', points: authored.map((point) => pointInWorld(posed.frames.get(point.frame)!, point.position)),
      }, values, missing && `${muscle.node_id} refers to unknown frame ${missing.frame}.`);
    }
  }

  const named = new Set(elements.map((element) => element.name));
  problems.push(...source.diagnostics.filter((item) => !item.node_ids?.[0] || !named.has(item.node_ids[0])).map((item) => item.message));
  const mechanism = address(source.node_path.slice(0, -1), mechanismId);
  return { groups: [{ key: elementKey(mechanism), address: mechanism, name: mechanismId, elements, problem: problems[0] }] };
}
