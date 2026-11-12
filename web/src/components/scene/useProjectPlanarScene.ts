import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import { fetchProjectPlanarScene } from '@/api/projectScene';
import { isHttpConflict } from '@/api/request';
import { studioRegionIssue } from '@/components/activity/StudioRegion';
import type { StudioPlanarSceneRequest } from '@/generated/studioContracts';
import { resolveProjectPlanarScene, type PlanarScenePose } from './planarModel';
import type { SceneView } from './sceneView';

export function useProjectPlanarSceneQuery(projectId: string, source: StudioPlanarSceneRequest) {
  return useQuery({
    queryKey: ['project-planar-scene', projectId, source],
    queryFn: ({ signal }) => fetchProjectPlanarScene(projectId, source, signal),
    refetchOnWindowFocus: false,
    retry: false,
    placeholderData: (previous) => previous?.project_id === projectId && previous.document_id === source.document_id &&
      JSON.stringify(previous.node_path) === JSON.stringify(source.node_path) ? previous : undefined,
  });
}

export function planarSceneFailure(error: unknown, retry: () => void, refreshProject: () => void): SceneView {
  return { kind: 'failed', issue: studioRegionIssue('scene', error), retry: isHttpConflict(error) ? refreshProject : retry };
}

export function useProjectPlanarScene(
  projectId: string, source: StudioPlanarSceneRequest, pose: PlanarScenePose, refreshProject: () => void,
): SceneView {
  const result = useProjectPlanarSceneQuery(projectId, source);
  const scene = useMemo(() => result.data && resolveProjectPlanarScene(result.data, pose), [result.data, pose]);
  if (result.isError) return planarSceneFailure(result.error, () => { void result.refetch(); }, refreshProject);
  return scene && !result.isPlaceholderData ? { kind: 'geometry', scene } : { kind: 'loading' };
}
