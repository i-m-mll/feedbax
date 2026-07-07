import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiRequestError } from '@/api/request';
import { fetchAnalysisPackages } from '@/api/analysisAPI';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('analysisAPI failure behavior', () => {
  it('throws a typed backend-unavailable error instead of returning stub packages', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => {
      throw new TypeError('connection refused');
    }));

    await expect(fetchAnalysisPackages()).rejects.toMatchObject({
      name: 'ApiRequestError',
      kind: 'network',
      path: '/api/analyses/packages',
    });
  });

  it('throws a typed contract error instead of returning stub packages on schema drift', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => Response.json({ data: { packages: 'not an array' } })));

    await expect(fetchAnalysisPackages()).rejects.toMatchObject({
      name: 'ApiRequestError',
      kind: 'contract',
      path: '/api/analyses/packages',
    });
    await expect(fetchAnalysisPackages()).rejects.toBeInstanceOf(ApiRequestError);
  });
});
