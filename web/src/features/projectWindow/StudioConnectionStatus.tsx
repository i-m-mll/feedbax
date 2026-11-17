import { useEffect, useState } from 'react';
import { connectionGraceMs, retryConnection, useConnection } from '@/api/connection';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';

export function StudioConnectionStatus() {
  const connection = useConnection();
  const since = connection.kind === 'connected' ? null : connection.since;
  const [shownAfter, setShownAfter] = useState<number | null>(null);
  useEffect(() => {
    if (since === null) return;
    const timer = window.setTimeout(() => setShownAfter(since), Math.max(0, since + connectionGraceMs - Date.now()));
    return () => window.clearTimeout(timer);
  }, [since]);
  if (connection.kind === 'connected' || shownAfter !== connection.since) return null;
  if (connection.kind === 'connecting') {
    return <span role="status" className="text-shell-muted">{studioName('connecting')}</span>;
  }
  return (
    <div role="alert" className="flex min-w-0 items-center gap-1">
      <span className="min-w-0 break-words">{studioReason('unreachable', studioName('service'))}</span>
      <Button onClick={retryConnection}>{studioName('retry')}</Button>
    </div>
  );
}
