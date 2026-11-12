import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchProjectFolder, patchProjectFolder, writeProjectPresentation } from '@/api/projectFolders';
import type { StudioProjectSnapshot } from '@/generated/studioContracts';

const revision = 'a'.repeat(64);
const presentationRevision = 'b'.repeat(64);

function snapshot(): StudioProjectSnapshot {
  return {
    project_id: 'reaches',
    revision,
    presentation_revision: presentationRevision,
    semantic_root: 'c'.repeat(64),
    presentation: {},
    file_revisions: [{ address: { kind: 'document', id: 'model' }, revision }],
    project: {
      documents: {
        model: {
          schema: 'feedbax.spec.graph',
          payload: JSON.parse('{"__proto__":{"named":1},"parameters":{"width":4}}'),
        },
      },
      steps: {
        train: {
          kind: 'train',
          base: { document: 'model' },
          sweep: {
            axes: { rate: { path: 'training.rate', values: [0.001, 0.002] } },
            rows: { reference: { from: 'baseline' } },
          },
        },
      },
    },
  };
}

afterEach(() => vi.unstubAllGlobals());

describe('Studio project folder HTTP contracts', () => {
  it('keeps authored aliases, omissions and JSON keys through a validated read', async () => {
    const response = snapshot();
    const fetch = vi.fn().mockResolvedValue(Response.json(response));
    vi.stubGlobal('fetch', fetch);
    const signal = new AbortController().signal;

    const read = await fetchProjectFolder('reaches', signal);
    expect(read.project.steps?.train.sweep?.rows?.reference?.from).toBe('baseline');
    expect(read.project.steps?.train).not.toHaveProperty('inputs');
    expect(Object.entries(read.project.documents?.model.payload as object)).toContainEqual(['__proto__', { named: 1 }]);
    expect(fetch).toHaveBeenCalledWith('/api/projects/reaches', expect.objectContaining({
      cache: 'no-store',
      signal,
    }));
  });

  it('rejects an unvalidated write operation before sending it', async () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    const edit = {
      revision,
      address: { kind: 'document' as const, id: 'model' },
      patches: [{ op: 'remove' as const, path: '/parameters/width', value: 8 }],
    };
    await expect(patchProjectFolder('reaches', edit)).rejects.toMatchObject({ kind: 'contract' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('rejects a malformed authored response and retains HTTP conflict details', async () => {
    const bad = snapshot();
    const train = bad.project.steps!.train;
    Object.assign(train.base, { unsupported: true });
    const detail = { message: 'The presentation changed', current: 'd'.repeat(64), resource: 'presentation' };
    const fetch = vi.fn()
      .mockResolvedValueOnce(Response.json(bad))
      .mockResolvedValueOnce(Response.json({ detail }, { status: 409 }));
    vi.stubGlobal('fetch', fetch);

    await expect(fetchProjectFolder('reaches')).rejects.toMatchObject({ kind: 'contract' });
    await expect(writeProjectPresentation('reaches', {
      revision: presentationRevision,
      presentation: { visible_groups: ['calibration'] },
    })).rejects.toMatchObject({ kind: 'http', status: 409, detail });
    expect(fetch).toHaveBeenLastCalledWith('/api/projects/reaches/presentation', expect.objectContaining({
      method: 'PUT',
      cache: 'no-store',
    }));
  });
});
