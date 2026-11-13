import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  createProjectFolder, openProjectFolder, patchProjectFolder, renameProjectFolder, replayProjectHistory, writeProjectPresentation,
} from '@/api/projectFolders';
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
    declaration: { revision, data: 'e30=', authored_revision: revision },
    file_revisions: [{ data: 'e30=', address: { kind: 'document', id: 'model' }, revision }],
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

function document(project = 'reaches', data = snapshot()) {
  return { configuration: { project, store: 'results' }, snapshot: data };
}

describe('Studio project folder HTTP contracts', () => {
  it('keeps authored aliases, omissions and JSON keys through a validated read', async () => {
    const response = document();
    const fetch = vi.fn().mockResolvedValue(Response.json(response));
    vi.stubGlobal('fetch', fetch);
    const signal = new AbortController().signal;

    const { snapshot: read } = await openProjectFolder('reaches', signal);
    expect(read.project.steps?.train.sweep?.rows?.reference?.from).toBe('baseline');
    expect(read.project.steps?.train).not.toHaveProperty('inputs');
    expect(Object.entries(read.project.documents?.model.payload as object)).toContainEqual(['__proto__', { named: 1 }]);
    expect(fetch).toHaveBeenCalledWith('/api/projects/reaches/open', expect.objectContaining({
      cache: 'no-store',
      signal,
    }));
  });

  it('uses the same validated envelope for New and Rename and refuses a mismatched folder', async () => {
    const renamed = document('renamed');
    const fetch = vi.fn()
      .mockResolvedValueOnce(Response.json(document(), { status: 201 }))
      .mockResolvedValueOnce(Response.json(renamed))
      .mockResolvedValueOnce(Response.json(document('renamed', { ...snapshot(), project_id: 'other' })));
    vi.stubGlobal('fetch', fetch);

    expect(await createProjectFolder()).toEqual(document());
    expect(fetch).toHaveBeenNthCalledWith(1, '/api/projects', expect.objectContaining({ method: 'POST' }));
    expect(await renameProjectFolder('reaches', { revision, name: 'renamed' })).toEqual(renamed);
    expect(fetch).toHaveBeenNthCalledWith(2, '/api/projects/reaches/rename', expect.objectContaining({
      method: 'POST', body: JSON.stringify({ revision, name: 'renamed' }),
    }));
    await expect(openProjectFolder('reaches')).rejects.toMatchObject({ kind: 'contract' });
  });

  it('replays one revision-bound authored history transaction', async () => {
    const fetch = vi.fn().mockResolvedValue(Response.json(snapshot()));
    vi.stubGlobal('fetch', fetch);
    const body = { revision, files: [{ address: { kind: 'document' as const, id: 'model' }, before: revision, after: 'e30=' }] };
    expect(await replayProjectHistory('reaches', body)).toEqual(snapshot());
    expect(fetch).toHaveBeenCalledWith('/api/projects/reaches/history/replay', expect.objectContaining({
      method: 'POST', body: JSON.stringify(body),
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
      .mockResolvedValueOnce(Response.json(document('reaches', bad)))
      .mockResolvedValueOnce(Response.json({ detail }, { status: 409 }));
    vi.stubGlobal('fetch', fetch);

    await expect(openProjectFolder('reaches')).rejects.toMatchObject({ kind: 'contract' });
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
