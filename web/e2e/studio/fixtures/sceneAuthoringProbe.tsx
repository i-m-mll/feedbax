import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useLayoutEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { SceneCanvas, SceneDrawer, SceneElementInspector } from '@/components/scene/SceneRegion';
import { ScenePlanarFields } from '@/components/scene/ScenePlanarFields';
import type { PlanarScenePose } from '@/components/scene/planarModel';
import { fittedCamera } from '@/components/scene/sceneView';
import { usePlanarSceneAuthoring } from '@/components/scene/usePlanarSceneAuthoring';
import { projectMechanicsAssembly } from '@/features/domains/mechanicsAssembly';
import { planPlanarOperation, readPlanarDraft, type PlanarOperation } from '@/features/domains/planarAuthoring';
import { applyAuthoredEdits } from '@/features/projectFolder/authoredEdits';
import { openProjectEditor } from '@/features/projectFolder/openProjectEditor';

const occurrence = { document_id: 'single', node_path: ['mechanics'] };
const overlays = new Set<never>();
type Inspection = Exclude<PlanarScenePose, { kind: 'recorded' }>;

export async function mountAuthoringProbe(container: HTMLElement, projectId: string) {
  const session = await openProjectEditor(projectId);
  const { editor } = session;
  const root = createRoot(container);
  const client = new QueryClient();
  let current: ReturnType<typeof usePlanarSceneAuthoring> | null = null;

  function Regions({ pose }: { pose: Inspection }) {
    const authoring = usePlanarSceneAuthoring(editor, occurrence, pose);
    const { view, beginAttachment, editable } = authoring;
    const [selectedId, onSelect] = useState<string | null>(null);
    const [camera, onCameraChange] = useState(fittedCamera);
    useLayoutEffect(() => { current = authoring; }, [authoring]);
    return (
      <>
        <div style={{ width: 840, height: 460 }}>
          <SceneCanvas {...{ view, selectedId, onSelect, camera, onCameraChange }}
            time={0} hiddenOverlays={overlays} beginAttachment={editable ? beginAttachment : undefined} />
        </div>
        {view.kind === 'geometry' && (
          <>
            <SceneDrawer scene={view.scene} {...{ selectedId, onSelect }} />
            <SceneElementInspector scene={view.scene} selectedId={selectedId} />
            <ScenePlanarFields authoring={authoring} selectedId={selectedId} />
          </>
        )}
      </>
    );
  }
  function show(pose: Inspection) {
    root.render(<QueryClientProvider client={client}><Regions pose={pose} /></QueryClientProvider>);
  }
  show({ kind: 'reference' });
  return {
    show,
    operate: (operation: PlanarOperation) => current!.apply(operation),
    expected: (operation: PlanarOperation) => {
      const { draft } = editor.getSnapshot();
      const planned = planPlanarOperation(draft, occurrence, operation);
      if (planned.ok === false) throw new Error(planned.reason);
      return applyAuthoredEdits(draft, { kind: 'document', id: occurrence.document_id }, planned.value.patches, true).project;
    },
    state() {
      const { draft, saved, pending, undoName, redoName } = editor.getSnapshot();
      const graph = readPlanarDraft(draft, occurrence);
      const view = current?.view;
      const scene = view?.kind === 'geometry' ? view.scene : null;
      return {
        draft, savedRevision: saved.revision, pending, undoName, redoName, scene,
        compiledIdentity: current?.source?.revision === saved.revision ? current.source.model?.model_identity : undefined,
        assembly: graph.ok ? projectMechanicsAssembly(graph.value.graph) : null,
        points: graph.ok ? graph.value.graph.nodes.cross_chain?.params?.path_points : null,
      };
    },
    undo: () => editor.undo(),
    redo: () => editor.redo(),
    flush: () => editor.flush(),
    close: () => { root.unmount(); session.close(); },
  };
}
