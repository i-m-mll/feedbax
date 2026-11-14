import { requestResponse } from '@/api/request';
import { parseContract } from '@/generated/studioContracts';

export async function reportStudioRenderFailure(name: string, error: Error, componentStack: string) {
  const report = parseContract('StudioRenderFailure', {
    region: name.slice(0, 80),
    error_type: error.name.slice(0, 80) || 'Error',
    detail: error.message.slice(0, 2048),
    component_stack: componentStack.slice(0, 4096),
  });
  await requestResponse('/api/studio/render-failure', {
    method: 'POST', body: JSON.stringify(report), keepalive: true,
  });
}
