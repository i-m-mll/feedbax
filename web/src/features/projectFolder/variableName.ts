/**
 * A dataset variable's name: a versioned semantic key (`/v1/<root>/<facet>/<owner…>/<symbol>`) reads as its owner and
 * symbol joined by dots, so `/v1/model/state/plant/effector~1velocity` is `plant.effector/velocity`; any other
 * variable is named by itself.
 */
export function variableName(key: string) {
  const parts = /^\/v1\/[^/]+\/[^/]+\/(.+)$/.exec(key)?.[1].split('/');
  return parts ? parts.map((token) => token.replace(/~1/g, '/').replace(/~0/g, '~')).join('.') : key;
}
