import { expect, test, type Page } from '@playwright/test';
import { execFile } from 'node:child_process';
import { readFile, rename, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import type { openProjectEditor } from '../../src/features/projectFolder/openProjectEditor';

const repo = fileURLToPath(new URL('../../../', import.meta.url));
const run = promisify(execFile);
const gainPath = '/payload/nodes/gain/params/gain';
type Probe = { folderEditor: Awaited<ReturnType<typeof openProjectEditor>> };

async function folderFixture(action: 'seed' | 'read', root: string) {
  return run('uv', ['run', '--no-sync', 'python', '-B', 'web/e2e/studio/fixtures/folder_project.py', action, root],
    { cwd: repo, timeout: 10_000 });
}

async function open(page: Page, id: string) {
  await page.evaluate(async (projectId) => {
    const entry = '/src/features/projectFolder/openProjectEditor.ts';
    const folder = await import(entry) as { openProjectEditor: typeof openProjectEditor };
    (window as unknown as Probe).folderEditor = await folder.openProjectEditor(projectId);
  }, id);
}

async function state(page: Page) {
  return page.evaluate(() => {
    const editor = (window as unknown as Probe).folderEditor.editor;
    const { saved, draft, outside, pending } = editor.getSnapshot();
    const gain = (project: typeof draft) => {
      const payload = project.documents?.model?.payload as { nodes: { gain: { params: { gain: number } } } };
      return payload.nodes.gain.params.gain;
    };
    return { saved: gain(saved.project), draft: gain(draft), outside: outside ? gain(outside.project) : null, pending, revision: saved.revision };
  });
}

async function outsideWrite(root: string, gain: number) {
  const path = join(root, 'documents', 'model.json');
  const doc = JSON.parse(await readFile(path, 'utf-8'));
  doc.payload.nodes.gain.params.gain = gain;
  const replacement = `${path}.outside`;
  await writeFile(replacement, JSON.stringify(doc));
  await rename(replacement, path);
}

test('M2 folder controller settles, reopens, observes disk changes, and resolves a stale edit', async ({ page }, info) => {
  const projectId = 'controller-reach';
  const root = join(String(info.config.metadata.home), 'state', 'web', 'projects', projectId);
  await folderFixture('seed', root);
  await page.goto('/');
  await open(page, projectId);
  expect((await state(page)).saved).toBe(1);
  await page.evaluate(({ path }) => {
    (window as unknown as Probe).folderEditor.editor.edit({ kind: 'document', id: 'model' }, 'Gain', [{ op: 'replace', path, value: 3 }]);
  }, { path: gainPath });
  await expect.poll(() => state(page)).toMatchObject({ saved: 3, draft: 3, pending: 0 });
  await page.reload();
  await open(page, projectId);
  expect((await state(page)).saved).toBe(3);

  await outsideWrite(root, 4);
  await expect.poll(() => state(page)).toMatchObject({ saved: 4, draft: 4, outside: null, pending: 0 });

  const routePath = `**/api/projects/${projectId}`;
  let competed = false;
  await page.route(routePath, async (route) => {
    if (route.request().method() === 'PATCH' && !competed) {
      competed = true;
      await outsideWrite(root, 5);
    }
    await route.continue();
  });
  const refused = await page.evaluate(async ({ path }) => {
    const editor = (window as unknown as Probe).folderEditor.editor;
    editor.edit({ kind: 'document', id: 'model' }, 'Gain', [{ op: 'replace', path, value: 6 }]);
    try { await editor.flush(); } catch (error) { return (error as { status?: number }).status; }
    return null;
  }, { path: gainPath });
  expect(refused).toBe(409);
  expect(competed).toBe(true);
  expect(await state(page)).toMatchObject({ saved: 4, draft: 6, outside: 5, pending: 1 });
  await page.unroute(routePath);

  const revisions = await page.evaluate(async () => {
    const editor = (window as unknown as Probe).folderEditor.editor;
    await editor.keep('edits');
    const saved = await editor.flush();
    editor.undo();
    const undone = await editor.flush();
    return { saved, undone };
  });
  expect(revisions.undone).not.toBe(revisions.saved);
  expect(await state(page)).toMatchObject({ saved: 5, draft: 5, outside: null, revision: revisions.undone });
  const { stdout } = await folderFixture('read', root);
  expect(JSON.parse(stdout)).toEqual({ gain: 5, revision: revisions.undone });
  await page.evaluate(() => (window as unknown as Probe).folderEditor.close());
});
