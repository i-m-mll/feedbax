import { once } from 'node:events';
import { createServer, request, type ServerResponse } from 'node:http';

export async function observeJobStreams(target: string) {
  const streams = new Map<ServerResponse, { id: number; projects: string[] }>();
  let nextStream = 0;
  const server = createServer((incoming, outgoing) => {
    const url = new URL(incoming.url!, target);
    const upstream = request(url, { method: incoming.method, headers: incoming.headers }, (reply) => {
      outgoing.writeHead(reply.statusCode ?? 502, reply.headers);
      if (url.pathname === '/api/events' && reply.headers['content-type']?.startsWith('text/event-stream')) {
        streams.set(outgoing, { id: ++nextStream, projects: url.searchParams.getAll('project') });
      }
      reply.once('close', () => {
        if (!reply.complete && !outgoing.writableEnded) outgoing.destroy();
      });
      reply.pipe(outgoing);
    });
    outgoing.once('close', () => {
      streams.delete(outgoing);
      upstream.destroy();
    });
    upstream.on('error', (error) => outgoing.destroy(error));
    incoming.pipe(upstream);
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('Expected an observer loopback port');
  return {
    origin: `http://127.0.0.1:${address.port}`,
    subscriptions: () => [...streams.values()],
    close: async () => {
      const closed = once(server, 'close');
      server.close();
      server.closeAllConnections();
      await closed;
    },
  };
}
