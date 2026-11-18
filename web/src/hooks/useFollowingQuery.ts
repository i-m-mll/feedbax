import { useEffect, useRef } from 'react';
import { useQuery, type QueryFunctionContext, type QueryKey, type UseQueryOptions } from '@tanstack/react-query';

type FollowingOptions<T, K extends QueryKey> = Omit<UseQueryOptions<T, Error, T, K>, 'queryFn'> & {
  queryFn: (context: QueryFunctionContext<K>) => Promise<T>;
};

/**
 * A query that follows `version`, such as the jobs event cursor: when the version moves, the query fetches again.
 *
 * The version is not part of the key. Keying on it would abandon the fetch in flight at every change, and while a run
 * streams events faster than the server answers, no fetch would ever finish. Here a fetch in flight is left to finish;
 * if the version moved after it started, one more follows.
 */
export function useFollowingQuery<T, K extends QueryKey>(options: FollowingOptions<T, K>, version: string) {
  const latest = useRef(version);
  latest.current = version;
  const fetched = useRef<string>();
  const { queryFn } = options;
  const query = useQuery<T, Error, T, K>({
    ...options,
    queryFn: (context) => {
      fetched.current = latest.current;
      return queryFn(context);
    },
  });
  const { isFetching, refetch } = query;
  const enabled = options.enabled !== false;
  useEffect(() => {
    if (!enabled || isFetching || fetched.current === undefined || fetched.current === version) return;
    void refetch({ cancelRefetch: false });
  }, [enabled, isFetching, refetch, version]);
  return query;
}
