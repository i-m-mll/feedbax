import { useQuery } from '@tanstack/react-query';
import { useMemo } from 'react';
import { fetchProjectPlanarScene } from '@/api/projectScene';
import { isHttpConflict } from '@/api/request';
import { studioRegionIssue } from '@/components/activity/StudioRegion';
import type { StudioPlanarSceneRequest } from '@/generated/studioContracts';
import { resolveProjectPlanarScene, type PlanarScenePose } from './planarModel';
import type { SceneView } from './sceneView';

export function useProjectPlanarScene(
  projectId: string, source: StudioPlanarSceneRequest, pose: PlanarScenePose, refreshProject: () => void,
): SceneView {
  const result = useQuery({
    queryKey: ['project-planar-scene', projectId, source],
    queryFn: ({ signal }) => fetchProjectPlanarScene(projectId, source, signal),
    refetchOnWindowFocus: false,
    retry: false,
  });
  const scene = useMemo(() => result.data && resolveProjectPlanarScene(result.data, pose), [result.data, pose]);
  if (result.isError) return {
    kind: 'failed', issue: studioRegionIssue('scene', result.error),
    retry: isHttpConflict(result.error) ? refreshProject : () => { void result.refetch(); },
  };
  return scene ? { kind: 'geometry', scene } : { kind: 'loading' };
}
