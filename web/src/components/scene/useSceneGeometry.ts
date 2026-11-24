import { useQueries } from '@tanstack/react-query';
import { useCallback } from 'react';
import { fetchProjectSceneGeometry } from '@/api/projectScene';
import type { StudioSceneGeometryRequest } from '@/generated/studioContracts';
import type { SceneModelGeometry, StateValues } from './trialScene';

/** The Model's geometry in one pose: being read, the answer, or why it could not be read. */
export type SceneGeometry =
  | { readonly kind: 'loading' }
  | { readonly kind: 'ready'; readonly geometry: SceneModelGeometry }
  | { readonly kind: 'failed'; readonly error: unknown; readonly retry: () => void };

/** The most poses one request asks for; a longer trial is read in windows of this many frames. */
const windowSize = 256;

/**
 * The Model's mechanisms in each of `poses`, read in as few requests as the window allows, so that a trial's
 * playback reads its frames once and then draws each from what was read. `at(index)` is the pose at that index;
 * `undefined` when there is no Model or nothing to pose.
 */
export function useSceneGeometry(projectId: string, source: Pick<StudioSceneGeometryRequest, 'revision' | 'document_id'> | null,
  poses: readonly StateValues[] | undefined) {
  const windows = source && poses?.length
    ? Array.from({ length: Math.ceil(poses.length / windowSize) }, (_, index) =>
      ({ ...source, poses: poses.slice(index * windowSize, (index + 1) * windowSize).map((pose) => ({ ...pose })) }))
    : [];
  const reads = useQueries({
    queries: windows.map((request) => ({
      queryKey: ['scene-geometry', projectId, request],
      queryFn: ({ signal }: { signal: AbortSignal }) => fetchProjectSceneGeometry(projectId, request, signal),
      staleTime: Infinity,
      refetchOnWindowFocus: false,
    })),
  });
  const at = useCallback((index: number): SceneGeometry | undefined => {
    const read = reads[Math.floor(index / windowSize)];
    if (!read) return undefined;
    if (read.error) return { kind: 'failed', error: read.error, retry: () => { void read.refetch(); } };
    const frame = read.data?.frames[index % windowSize];
    return read.data && frame ? { kind: 'ready', geometry: { document_id: read.data.document_id, ...frame } } : { kind: 'loading' };
  }, [reads]);
  return windows.length ? at : undefined;
}
