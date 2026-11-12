import { expect, test } from '@playwright/test';
import type {
  ReportDocument, StudioJobSnapshot, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('authors an ordered managed Report from a planned Analyze figure', async ({ page, request }, info) => {
  const home = String(info.config.metadata.home);
  const module = 'web.e2e.studio.fixtures.managed_report';
  const project = JSON.parse(await runProjectFixture(module, 'seed', home, 'managed-report'));
  const path = '/api/projects/' + project;
  const mounted = '/?project=' + encodeURIComponent(project);
  const figures = page.getByRole('tabpanel', { name: 'Figures', exact: true });
  const document = page.getByRole('tabpanel', { name: 'Document', exact: true });
  const blocks = document.getByRole('listitem');
  const report = (saved: StudioProjectSnapshot) => saved.project.documents?.report.payload as unknown as ReportDocument;
  const inputs = (saved: StudioProjectSnapshot) => saved.project.steps?.report.inputs;
  const profile = { step: 'analyze', row: 'default', element: 'profile', product: 'figure' };
  const comparison = { ...profile, element: 'comparison' };
  const requests = async (revision: string) => JSON.parse(await runProjectFixture(module, 'requests', home, project, revision));

  async function snapshot() {
    const response = await request.get(path + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectDocument).snapshot;
  }
  async function windowMap(revision: string) {
    const response = await request.get(path + '/train-runs', { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!;
  }
  async function edit(action: () => Promise<unknown>) {
    const patched = page.waitForResponse((response) =>
      response.request().method() === 'PATCH' && new URL(response.url()).pathname === path);
    await action();
    const response = await patched;
    expect(response.ok(), await response.text()).toBe(true);
    await expect(page.getByText('Edited', { exact: true })).toBeHidden();
    return await response.json() as StudioProjectSnapshot;
  }
  async function palette(name: string) {
    await page.keyboard.press('Meta+K');
    const dialog = page.getByRole('dialog', { name: 'Command Palette', exact: true });
    await dialog.getByRole('combobox').fill(name);
    await dialog.getByRole('option', { name, exact: true }).press('Enter');
  }
  async function addFigure(name: string) {
    await document.getByRole('button', { name: 'Add Figure', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Add Figure', exact: true });
    await dialog.getByRole('combobox', { name: 'Figures' }).selectOption({ label: 'analyze · ' + name });
    return edit(() => dialog.getByRole('button', { name: 'Add Figure', exact: true }).click());
  }

  await page.goto(mounted);
  await mapStep(page, 'analyze').click();
  await page.getByRole('tab', { name: 'Figures', exact: true }).click();
  await figures.getByRole('button', { name: 'profile', exact: true }).click();
  await palette('Report');
  await expect(document.getByRole('textbox', { name: 'Report Title' })).toHaveValue('Profile');
  const created = await snapshot();
  expect(report(created)).toMatchObject({ completeness: 'complete', blocks: [{ figure: 'figure-profile' }] });
  expect(inputs(created)).toMatchObject({ 'figure-profile': profile });
  const map = await windowMap(created.revision);
  expect(map.problems.filter(({ step }) => step === 'report')).toEqual([]);
  expect(map.rows).toContainEqual(expect.objectContaining({
    key: expect.objectContaining({ step: 'report' }), status: 'not_run',
  }));
  expect(map.edges).toContainEqual(expect.objectContaining({
    from_key: { variant: 'default', step: 'analyze', row: 'default' },
    to_key: { variant: 'default', step: 'report', row: 'default' },
  }));
  const original = await requests(created.revision);

  await expect(document.getByRole('button', { name: 'Add Text', exact: true })).toBeEnabled();
  await palette('Add Text');
  const addText = page.getByRole('dialog', { name: 'Add Text', exact: true });
  await addText.getByRole('textbox', { name: 'Text' }).fill('**Velocity** by condition.');
  await edit(() => addText.getByRole('button', { name: 'Add Text', exact: true }).click());
  const ordered = await edit(() => blocks.nth(1).getByRole('button', { name: 'Move Earlier 2' }).click());
  expect(report(ordered).blocks).toEqual([{ markdown: '**Velocity** by condition.' }, { figure: 'figure-profile' }]);

  const repeated = await addFigure('profile');
  expect(report(repeated).blocks.slice(1)).toEqual([{ figure: 'figure-profile' }, { figure: 'figure-profile' }]);
  expect(inputs(repeated)).toEqual(inputs(ordered));
  const additional = await addFigure('comparison');
  expect(Object.values(inputs(additional)!)).toContainEqual(comparison);
  const removed = await edit(() => blocks.last().getByRole('button', { name: 'Remove 4' }).click());
  expect(inputs(removed)).toEqual(inputs(ordered));
  expect(report(removed).blocks).toEqual(report(repeated).blocks);

  await document.getByRole('textbox', { name: 'Report Title' }).focus();
  await page.getByRole('tab', { name: 'Document', exact: true }).focus();
  const undone = await edit(() => page.keyboard.press('Meta+Z'));
  expect(report(undone)).toEqual(report(additional));
  expect(inputs(undone)).toEqual(inputs(additional));
  const redone = await edit(() => page.keyboard.press('Meta+Shift+Z'));
  expect(report(redone)).toEqual(report(removed));
  expect(inputs(redone)).toEqual(inputs(removed));

  const permissive = await edit(() => document.getByRole('combobox', { name: 'Completeness' }).selectOption('allow_omissions'));
  const changed = await requests(permissive.revision);
  expect(changed.report).not.toBe(original.report);
  expect(changed.upstream).toEqual(original.upstream);
  const stale = await request.patch(path, { data: {
    revision: created.revision,
    files: [
      { address: { kind: 'document', id: 'report' }, patches: [
        { op: 'replace', path: '/payload/blocks', value: [{ figure: 'stale' }] },
      ] },
      { address: { kind: 'step', id: 'report' }, patches: [
        { op: 'replace', path: '/inputs', value: { stale: comparison } },
      ] },
    ],
  } });
  expect(stale.status()).toBe(409);

  await page.reload();
  await mapStep(page, 'report').click();
  await page.getByRole('tab', { name: 'Document', exact: true }).click();
  await expect(document.getByRole('combobox', { name: 'Completeness' })).toHaveValue('allow_omissions');
  await expect(blocks).toHaveCount(3);
  const reopened = await snapshot();
  expect(reopened.revision).toBe(permissive.revision);
  expect(report(reopened)).toEqual(report(permissive));
  expect(inputs(reopened)).toEqual(inputs(permissive));
  expect((await windowMap(reopened.revision)).problems.filter(({ step }) => step === 'report')).toEqual([]);
  const jobs = await request.get('/api/jobs', { params: { project } });
  expect(jobs.ok(), await jobs.text()).toBe(true);
  expect((await jobs.json() as StudioJobSnapshot).jobs).toEqual([]);
});
