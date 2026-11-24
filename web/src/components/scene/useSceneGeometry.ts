import { useQueryClient } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { fetchProjectSceneGeometry } from '@/api/projectScene';
import type { StudioSceneGeometryRequest } from '@/generated/studioContracts';
import type { SceneModelGeometry, StateValues } from './trialScene';

/** The Model's geometry in one pose, as last read: nothing yet, the answer, or why it could not be read. */
export type SceneGeometry =
  | { readonly kind: 'loading' }
  | { readonly kind: 'ready'; readonly geometry: SceneModelGeometry }
  | { readonly kind: 'failed'; readonly error: unknown; readonly retry: () => void };

type Read = { readonly key: string; readonly source: string; readonly result: SceneGeometry };

/**
 * The Model's mechanisms posed by `values`, reconciled toward the latest pose asked for: one read is in flight at a
 * time, and the last pose read for the same Model stays drawn while the next is read, so a playing clock draws every
 * pose the service answers in time for and is never left without one.
 */
export function useSceneGeometry(projectId: string, source: Pick<StudioSceneGeometryRequest, 'revision' | 'document_id'> | null, values: StateValues | undefined): SceneGeometry | undefined {
  const client = useQueryClient();
  const model = source && JSON.stringify([projectId, source.document_id, source.revision]);
  const request = source && values ? { ...source, poses: [values] } : null;
  const key = request && JSON.stringify([projectId, request]);
  const [read, setRead] = useState<Read>();
  const [inFlight, setInFlight] = useState(false);
  useEffect(() => {
    if (!request || !key || !model || inFlight || read?.key === key) return;
    setInFlight(true);
    void client.fetchQuery({
      queryKey: ['scene-geometry', projectId, request],
      queryFn: ({ signal }) => fetchProjectSceneGeometry(projectId, request, signal),
      staleTime: Infinity,
    }).then(
      ({ document_id, frames }): SceneGeometry => ({
        kind: 'ready', geometry: { document_id, ...frames[0] },
      }),
      (error): SceneGeometry => ({ kind: 'failed', error, retry: () => setRead(undefined) }),
    ).then((result) => {
      setRead({ key, source: model, result });
      setInFlight(false);
    });
  }, [client, projectId, request, key, model, inFlight, read]);
  if (!request) return undefined;
  return read && read.source === model ? read.result : { kind: 'loading' };
}
