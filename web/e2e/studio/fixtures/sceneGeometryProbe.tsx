import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useLayoutEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { SceneCanvas, SceneDrawer, SceneElementInspector } from '@/components/scene/SceneRegion';
import type { PlanarScenePose } from '@/components/scene/planarModel';
import { fittedCamera, type SceneView } from '@/components/scene/sceneView';
import { useProjectPlanarScene } from '@/components/scene/useProjectPlanarScene';
import type { StudioPlanarSceneRequest } from '@/generated/studioContracts';

const hiddenOverlays = new Set<never>();

export function mountSceneProbe(container: HTMLElement, project: string) {
  const root = createRoot(container);
  const queryClient = new QueryClient();
  let current: { version: number; view: SceneView } | undefined;
  let version = 0;

  function Region({ source, pose, requestVersion }: {
    source: StudioPlanarSceneRequest; pose: PlanarScenePose; requestVersion: number;
  }) {
    const view = useProjectPlanarScene(project, source, pose, () => {});
    const [selectedId, onSelect] = useState<string | null>(null);
    const [camera, onCameraChange] = useState(fittedCamera);
    useLayoutEffect(() => { current = { version: requestVersion, view }; }, [requestVersion, view]);
    return (
      <>
        <div style={{ width: 860, height: 520 }}>
          <SceneCanvas {...{ view, camera, onCameraChange, selectedId, onSelect, hiddenOverlays }} time={0} />
        </div>
        {view.kind === 'geometry' && (
          <>
            <SceneDrawer scene={view.scene} {...{ selectedId, onSelect }} />
            <SceneElementInspector scene={view.scene} selectedId={selectedId} />
          </>
        )}
      </>
    );
  }

  return {
    show(source: StudioPlanarSceneRequest, pose: PlanarScenePose) {
      version += 1;
      root.render(
        <QueryClientProvider client={queryClient}>
          <Region source={source} pose={pose} requestVersion={version} />
        </QueryClientProvider>,
      );
      return version;
    },
    current: () => current,
    close: () => root.unmount(),
  };
}
