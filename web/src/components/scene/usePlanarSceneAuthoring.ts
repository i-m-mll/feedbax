import { useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { attachmentPoints, commitPlanarOperation, fixedPath, readPlanarDraft, type PlanarOccurrence, type PlanarOperation } from '@/features/domains/planarAuthoring';
import type { ProjectEditController } from '@/features/projectFolder/projectEditController';
import type { ResolvedSceneAnchor } from '@/features/scenario/projections';
import type { AcausalGraphSpec } from '@/generated/studioContracts';
import { frameLocalDelta, posePlanarPreview, resolveProjectPlanarScene, type PlanarScenePose } from './planarModel';
import { planarDraftGeometry } from './planarDraftGeometry';
import type { ScenePoint } from './planarProjection';
import type { SceneView } from './sceneView';
import { planarSceneFailure, useProjectPlanarSceneQuery } from './useProjectPlanarScene';

export interface SceneAttachmentDrag {
  isActive(): boolean;
  move(worldDelta: ScenePoint): void;
  commit(): void;
  cancel(): void;
}

type AttachmentEdit = Extract<PlanarOperation, { kind: 'set-attachment' }>;
type EditablePose = Exclude<PlanarScenePose, { kind: 'recorded' }>;

export function usePlanarSceneAuthoring(editor: ProjectEditController, occurrence: PlanarOccurrence, pose: EditablePose) {
  const snapshot = useSyncExternalStore(editor.subscribe, editor.getSnapshot);
  const { saved, draft, outside } = snapshot;
  const query = useProjectPlanarSceneQuery(saved.project_id, {
    document_id: occurrence.document_id, node_path: [...occurrence.node_path], revision: saved.revision,
  });
  const source = query.data;
  const savedDraft = useMemo(() => readPlanarDraft(saved.project, occurrence), [saved.project, occurrence]);
  const currentDraft = useMemo(() => readPlanarDraft(draft, occurrence), [draft, occurrence]);
  const previous = useRef<{ revision: string; occurrence: string; graph: AcausalGraphSpec }>();
  const [gesture, setGesture] = useState<{ base: string; scope: string; edit: AttachmentEdit } | null>(null);
  const active = useRef<symbol>();
  const [editIssue, setEditIssue] = useState<string | null>(null);

  const occurrenceKey = JSON.stringify([saved.project_id, occurrence.document_id, occurrence.node_path]);
  useLayoutEffect(() => {
    if (source?.revision === saved.revision && savedDraft.ok) {
      previous.current = { revision: saved.revision, occurrence: occurrenceKey, graph: savedDraft.value.graph };
    }
  }, [source, saved.revision, savedDraft, occurrenceKey]);
  const authored = currentDraft.ok ? currentDraft.value.graph : null;
  const fingerprint = useMemo(() => authored && JSON.stringify(authored), [authored]);
  const scope = JSON.stringify([occurrenceKey, pose]);
  const live = useRef({ scope, editor, fingerprint, outside });
  useLayoutEffect(() => { live.current = { scope, editor, fingerprint, outside }; }, [scope, editor, fingerprint, outside]);
  const basis = source?.revision === saved.revision && savedDraft.ok ? savedDraft.value.graph
    : previous.current?.revision === source?.revision && previous.current?.occurrence === occurrenceKey ? previous.current.graph : undefined;
  const currentGesture = !outside && gesture?.base === fingerprint && gesture.scope === scope ? gesture.edit : undefined;
  const geometry = useMemo(() => source?.model && basis && authored && planarDraftGeometry(source.model, basis, authored, currentGesture),
    [source, basis, authored, currentGesture]);

  let view: SceneView = { kind: 'loading' };
  if (query.isError) view = planarSceneFailure(query.error, () => { void query.refetch(); }, () => editor.connected());
  else if (source && (geometry || (!source.model && !query.isPlaceholderData))) {
    view = { kind: 'geometry', scene: resolveProjectPlanarScene(source, pose, geometry || undefined) };
  }

  function apply(operation: PlanarOperation) {
    const result = commitPlanarOperation(editor, occurrence, operation);
    setEditIssue(result.ok === false ? result.reason : null);
    return result;
  }

  function beginAttachment(anchor: ResolvedSceneAnchor): SceneAttachmentDrag | null {
    const { muscle, point_index: index } = anchor.metadata;
    if (outside || active.current || !authored || !fingerprint || !geometry || typeof muscle !== 'string' || typeof index !== 'number') return null;
    const node = authored.nodes?.[muscle];
    const point = node && fixedPath(node) ? attachmentPoints(node)?.[index] : null;
    const posed = posePlanarPreview(geometry, pose);
    const frame = point && posed.ok ? posed.frames.get(point.frame) : undefined;
    if (!point || !frame) return null;
    const initial = fingerprint;
    const id = Symbol();
    active.current = id;
    setEditIssue(null);
    let edit: AttachmentEdit | undefined;
    function end() {
      if (active.current !== id) return false;
      active.current = undefined;
      setGesture(null);
      return true;
    }
    function isActive() {
      return active.current === id && live.current.scope === scope && live.current.editor === editor &&
        live.current.fingerprint === initial && !live.current.outside;
    }
    return {
      isActive,
      move(delta) {
        if (!isActive()) { end(); return; }
        const [x, y] = frameLocalDelta(frame, delta);
        const offset: ScenePoint = [point.offset[0] + x, point.offset[1] + y];
        if (!offset.every(Number.isFinite)) return;
        edit = { kind: 'set-attachment', node: muscle, index, expectedFrame: point.frame, attachment: { frame: point.frame, offset } };
        setGesture({ base: initial, scope, edit });
      },
      commit() {
        if (!isActive()) { end(); return; }
        if (!end() || !edit) return;
        const current = readPlanarDraft(editor.getSnapshot().draft, occurrence);
        if (live.current.scope !== scope || live.current.editor !== editor || current.ok === false || JSON.stringify(current.value.graph) !== initial) {
          setEditIssue('The authored model changed during the gesture. Choose the attachment again.');
          return;
        }
        apply(edit);
      },
      cancel: end,
    };
  }

  return { view, source, authored, apply, editIssue, beginAttachment, editable: Boolean(geometry && !outside) };
}
