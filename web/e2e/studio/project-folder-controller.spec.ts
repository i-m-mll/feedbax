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
type Probe = {
  folderEditor: Awaited<ReturnType<typeof openProjectEditor>>;
  renameCommand?: ReturnType<Probe['folderEditor']['editor']['rename']>;
};

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
    const { saved, draft, outside, pending, configuration, undoName } = editor.getSnapshot();
    const gain = (project: typeof draft) => {
      const payload = project.documents?.model?.payload as { nodes: { gain: { params: { gain: number } } } };
      return payload.nodes.gain.params.gain;
    };
    return {
      saved: gain(saved.project), draft: gain(draft), outside: outside ? gain(outside.project) : null,
      pending, revision: saved.revision, id: saved.project_id, name: configuration.project, undoName,
    };
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

test('M2 managed commands preserve edits, identity and names across Rename and reopen', async ({ page, request }, info) => {
  const projects = join(String(info.config.metadata.home), 'state', 'web', 'projects');
  const projectId = 'commands-reach';
  const root = join(projects, projectId);
  await folderFixture('seed', root);
  await page.goto('/');

  const blank = await page.evaluate(async () => {
    const entry = '/src/features/projectFolder/openProjectEditor.ts';
    const folder = await import(entry) as typeof import('../../src/features/projectFolder/openProjectEditor');
    const session = await folder.newProjectEditor();
    const original = session.editor.getSnapshot();
    const renamed = await session.editor.rename('blank-name');
    session.close();
    const reopened = await folder.openProjectEditor(original.saved.project_id);
    const after = reopened.editor.getSnapshot();
    reopened.close();
    return { original: original.saved, renamed, after: { configuration: after.configuration, snapshot: after.saved } };
  });
  expect(blank.renamed.configuration.project).toBe('blank-name');
  expect(blank.after).toEqual(blank.renamed);
  expect(blank.renamed.snapshot).toMatchObject({
    project_id: blank.original.project_id, file_revisions: [], semantic_root: blank.original.semantic_root,
  });
  expect(JSON.parse(await readFile(join(projects, blank.original.project_id, 'feedbax.project.json'), 'utf8')))
    .toMatchObject({ project: 'blank-name' });
  expect(JSON.parse(await readFile(join(projects, blank.original.project_id, 'presentation.json'), 'utf8'))).toEqual({});

  await open(page, projectId);
  const renamePath = `**/api/projects/${projectId}/rename`;
  const renameSeen = page.waitForRequest(renamePath);
  let releaseRename!: () => void;
  const renameGate = new Promise<void>((resolve) => { releaseRename = resolve; });
  await page.route(renamePath, async (route) => { await renameGate; await route.continue(); });
  try {
    await page.evaluate((path) => {
      const probe = window as unknown as Probe;
      probe.folderEditor.editor.edit({ kind: 'document', id: 'model' }, 'Before Rename', [{ op: 'replace', path, value: 3 }]);
      probe.renameCommand = probe.folderEditor.editor.rename('renamed');
    }, gainPath);
    const sent = await renameSeen;
    const before = await state(page);
    expect(before).toMatchObject({ saved: 3, pending: 0 });
    expect(sent.postDataJSON()).toEqual({ name: 'renamed', revision: before.revision });
    await page.evaluate((path) => {
      (window as unknown as Probe).folderEditor.editor.edit(
        { kind: 'document', id: 'model' }, 'During Rename', [{ op: 'replace', path, value: 4 }],
      );
    }, gainPath);
  } finally {
    releaseRename();
  }
  await page.evaluate(async () => {
    const probe = window as unknown as Probe;
    await probe.renameCommand;
    await probe.folderEditor.editor.flush();
  });
  await page.unroute(renamePath);
  expect(await state(page)).toMatchObject({ id: projectId, name: 'renamed', saved: 4, pending: 0, undoName: 'During Rename' });

  await page.route(renamePath, async (route) => {
    const { revision } = route.request().postDataJSON() as { revision: string };
    const competing = await request.post(`/api/projects/${projectId}/rename`, {
      data: { name: 'competing-name', revision },
    });
    expect(competing.ok()).toBe(true);
    await route.continue();
  });
  const refused = await page.evaluate(async () => {
    try { await (window as unknown as Probe).folderEditor.editor.rename('stale-name'); }
    catch (error) { return (error as { status?: number }).status; }
    return null;
  });
  expect(refused).toBe(409);
  expect(await state(page)).toMatchObject({ name: 'competing-name', saved: 4, pending: 0, undoName: 'During Rename' });
  await page.unroute(renamePath);

  let requests = 0;
  await page.route(renamePath, async (route) => {
    requests++;
    const response = await route.fetch();
    expect(response.ok()).toBe(true);
    await route.abort('failed');
  });
  const recoveredName = await page.evaluate(async () => {
    const result = await (window as unknown as Probe).folderEditor.editor.rename('ack-lost');
    return result.configuration.project;
  });
  expect(recoveredName).toBe('ack-lost');
  expect(requests).toBe(1);
  await page.unroute(renamePath);

  const openPath = `**/api/projects/${projectId}/open`;
  let seenOpen!: () => void;
  const openSeen = new Promise<void>((resolve) => { seenOpen = resolve; });
  let releaseOpen!: () => void;
  const openGate = new Promise<void>((resolve) => { releaseOpen = resolve; });
  let held = false;
  await page.route(openPath, async (route) => {
    const response = await route.fetch();
    if (!held && (await response.json()).configuration.project === 'outside-name') {
      held = true;
      seenOpen();
      await openGate;
    }
    await route.fulfill({ response });
  });
  try {
    const outside = await request.post(`/api/projects/${projectId}/rename`, {
      data: { name: 'outside-name', revision: (await state(page)).revision },
    });
    expect(outside.ok()).toBe(true);
    await openSeen;
    await page.evaluate((path) => {
      (window as unknown as Probe).folderEditor.editor.edit(
        { kind: 'document', id: 'model' }, 'During Open', [{ op: 'replace', path, value: 5 }],
      );
    }, gainPath);
  } finally {
    releaseOpen();
  }
  await page.evaluate(() => (window as unknown as Probe).folderEditor.editor.flush());
  await page.unroute(openPath);
  expect(await state(page)).toMatchObject({ name: 'outside-name', saved: 5, pending: 0, outside: null });

  const revision = await page.evaluate(async () => {
    const { editor, close } = (window as unknown as Probe).folderEditor;
    editor.undo();
    const revision = await editor.flush();
    close();
    return revision;
  });
  const { stdout } = await folderFixture('read', root);
  expect(JSON.parse(stdout)).toEqual({ gain: 4, revision });
  await page.reload();
  await open(page, projectId);
  expect(await state(page)).toMatchObject({ id: projectId, name: 'outside-name', saved: 4, revision });
  await page.evaluate(() => (window as unknown as Probe).folderEditor.close());
});
