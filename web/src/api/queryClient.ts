import { onlineManager, QueryClient } from '@tanstack/react-query';
import { getConnection, subscribeConnection } from './connection';
import { isUnanswered } from './request';

// Queries pause while the service is unreachable and resume when the connection returns,
// so a region keeps its last content, or its loading state, through an outage.
onlineManager.setEventListener((setOnline) => {
  const update = () => setOnline(getConnection().kind !== 'unreachable');
  update();
  return subscribeConnection(update);
});

export function createStudioQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: (_count, error) => isUnanswered(error),
        retryDelay: 500,
      },
      mutations: { networkMode: 'always', retry: false },
    },
  });
}
