import { attachmentPoints, type PlanarOperation } from '@/features/domains/planarAuthoring';
import type { AcausalGraphSpec, ScenePlanarModel } from '@/generated/studioContracts';
import type { PlanarGeometry } from './planarModel';

type AttachmentEdit = Extract<PlanarOperation, { kind: 'set-attachment' }>;

function changedNumber(previous: unknown, next: unknown, prepared: number) {
  if (previous === next) return prepared;
  return typeof next === 'number' && Number.isFinite(next) ? next : null;
}

function offset(value: unknown): [number, number] | null {
  if (value === undefined) return [0, 0];
  return Array.isArray(value) && value.length === 2 && value.every(Number.isFinite) ? [value[0], value[1]] : null;
}

export function planarDraftGeometry(model: ScenePlanarModel, before: AcausalGraphSpec, draft: AcausalGraphSpec, gesture?: AttachmentEdit): PlanarGeometry | null {
  const sourceNodes = before.nodes ?? {};
  const nodes = draft.nodes ?? {};
  const relevant = (type: string) => ['PlanarLink', 'RevoluteJoint', 'MusclePath', 'Anchor', 'WorldFrame'].includes(type.replace(/^feedbax\./, ''));
  const sourceIds = Object.keys(sourceNodes).filter((id) => relevant(sourceNodes[id].type));
  const draftIds = Object.keys(nodes).filter((id) => relevant(nodes[id].type));
  if (sourceIds.length !== draftIds.length || sourceIds.some((id) => nodes[id]?.type !== sourceNodes[id].type)) return null;

  const distalLengthChanges = new Map<string, number>();
  const links: PlanarGeometry['links'] = [];
  for (const binding of model.links) {
    const { body_id: body, joint_id: joint } = binding;
    const previous = sourceNodes[joint]?.params;
    const next = nodes[joint]?.params;
    if (!previous || !next || next.parent_frame !== binding.parent_frame || next.child_frame !== binding.child_frame) return null;
    const length = changedNumber(sourceNodes[body]?.params?.length, nodes[body]?.params?.length, binding.length);
    const angle = changedNumber(previous.frame_angle, next.frame_angle, binding.reference_angle);
    if (length === null || angle === null) return null;
    const oldOffset = offset(previous.frame_offset);
    const newOffset = offset(next.frame_offset);
    if (!oldOffset || !newOffset) return null;
    const [oldX, oldY] = oldOffset;
    const [newX, newY] = newOffset;
    const distal = `${body}.distal`;
    links.push({
      ...binding, length, reference_angle: angle,
      parent_pivot: [binding.parent_pivot[0] + (distalLengthChanges.get(binding.parent_frame) ?? 0) + newX - oldX,
        binding.parent_pivot[1] + newY - oldY],
      body_pivot: [binding.body_pivot[0] + (binding.child_frame === distal ? length - binding.length : 0), binding.body_pivot[1]],
    });
    distalLengthChanges.set(distal, length - binding.length);
  }

  const muscles: PlanarGeometry['muscles'] = [];
  for (const binding of model.muscles) {
    const node = nodes[binding.node_id];
    if (!node) return null;
    if (!binding.points) { muscles.push(binding); continue; }
    const points = attachmentPoints(node);
    if (!points) return null;
    muscles.push({ node_id: binding.node_id, points: points.map((point, index) => {
      const attachment = gesture?.node === binding.node_id && gesture.index === index ? gesture.attachment : point;
      return { frame: attachment.frame, position: attachment.offset };
    }) });
  }
  return { world_frame: model.world_frame, reference_qpos: model.reference_qpos, links, muscles };
}
