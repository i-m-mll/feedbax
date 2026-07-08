import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiRequestError } from '@/api/request';
import { dryRunAnalysisBundle, fetchAnalysisPackages } from '@/api/analysisAPI';

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

describe('analysis bundle dry-run API', () => {
  it('posts bundle and selection payloads and parses the dry-run response', async () => {
    const fetchMock = vi.fn(async () =>
      Response.json({
        schema_id: 'feedbax.spec.studio.api_transport',
        schema_version: 'feedbax.spec.studio.api_transport.v2',
        data: {
          schema_id: 'feedbax.spec.studio.api_transport',
          schema_version: 'feedbax.spec.studio.api_transport.v2',
          dry_run: {
            bundle_name: 'bundle-a',
            match_preview: {
              selection_spec: {
                schema_id: 'feedbax.spec.selection',
                schema_version: 'feedbax.spec.selection.v2',
                mode: 'explicit',
                manifest_kind: 'EvaluationRunManifest',
                ids: ['eval-a'],
                frozen_refs: [],
                metadata: {},
              },
              match_count: 1,
              parent_refs: [
                {
                  kind: 'EvaluationRunManifest',
                  id: 'eval-a',
                  role: 'evaluation_run',
                  metadata: {},
                },
              ],
              truncated: false,
            },
            matched_run_ids: ['eval-a'],
            stages: [
              {
                name: 'analysis',
                kind: 'analysis',
                status: 'would_run',
                depends_on: [],
                inputs: [],
                outputs: [
                  {
                    role: 'manifest',
                    required: true,
                    status: 'would_run',
                  },
                ],
                missing_roles: [],
              },
            ],
            metadata: {},
          },
        },
      })
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await dryRunAnalysisBundle({
      bundle: {
        name: 'bundle-a',
        stages: [{ name: 'analysis', kind: 'analysis', analysis_type: 'type-a' }],
      },
      selectionSpec: {
        mode: 'explicit',
        manifest_kind: 'EvaluationRunManifest',
        ids: ['eval-a'],
        frozen_refs: [],
        metadata: {},
      },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/api/analyses/bundles/dry-run',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('"selection_spec"'),
      })
    );
    expect(result.stages[0]).toMatchObject({ name: 'analysis', status: 'would_run' });
  });
});
