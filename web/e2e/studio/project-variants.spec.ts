import { expect, test } from '@playwright/test';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { enter, fixture, mapStep } from './fixtures/canonicalTrain';

test.use({
  trace: 'off',
  launchOptions: { args: ['--use-mock-keychain', '--password-store=basic'] },
  viewport: { width: 1440, height: 900 },
});

test('a Model value edit changes only its authored variant and keeps exact shared Train results', async ({ page, request }, info) => {
  test.setTimeout(55_000);
  const id = await fixture('seed-variants', String(info.config.metadata.home), 'model-variants');
  const projectPath = `/api/projects/${id}`;
  const trainPath = `${projectPath}/train-runs`;
  async function snapshot(): Promise<StudioProjectSnapshot> {
    const response = await request.get(projectPath);
    expect(response.ok(), await response.text()).toBe(true);
    return response.json();
  }
  async function rows(revision: string) {
    const response = await request.get(`${trainPath}?revision=${revision}&window_map=true`);
    expect(response.ok(), await response.text()).toBe(true);
    const plan = await response.json() as StudioSavedTrainRuns;
    expect(plan.window_map?.variants).toEqual(['base', 'compare', 'control']);
    expect(plan.rows.map(({ key }) => key)).toEqual(['base', 'compare', 'control'].map((variant) =>
      ({ variant, step: 'train', row: 'default' })));
    return Object.fromEntries(plan.rows.map((row) => [row.key.variant, row]));
  }

  const before = await snapshot();
  const url = `/?project=${encodeURIComponent(id)}`;
  const variants = page.getByRole('combobox', { name: 'Variant', exact: true });
  const runs = page.getByRole('table', { name: 'Runs' });
  const mass = page.getByRole('complementary', { name: 'Inspector' }).getByRole('spinbutton', { name: 'Edit Mass' });
  const plant = page.getByRole('complementary', { name: 'Details Drawer' }).getByRole('button', { name: 'plant', exact: true });
  await page.goto(url);
  await expect(variants).toHaveValue('base');
  await mapStep(page, 'train').click();
  const admission = page.waitForResponse((response) => response.request().method() === 'POST'
    && new URL(response.url()).pathname === trainPath);
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  const accepted = await admission;
  expect(accepted.status(), await accepted.text()).toBe(202);
  expect(accepted.request().postDataJSON().run_keys).toEqual([{ variant: 'base', step: 'train', row: 'default' }]);
  await expect(runs.getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  const initial = await rows(before.revision);
  expect(initial.base.result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
  for (const variant of ['compare', 'control']) {
    expect(initial[variant].request_id).toBe(initial.base.request_id);
    expect(initial[variant].result).toEqual(initial.base.result);
  }

  await variants.selectOption('compare');
  await expect(runs.getByText('Done', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Back', exact: true }).click();
  await expect(variants).toHaveValue('base');
  await page.getByRole('button', { name: 'Forward', exact: true }).click();
  await expect(variants).toHaveValue('compare');

  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  await page.getByRole('navigation', { name: 'Details', exact: true }).getByRole('tab', { name: 'Model', exact: true }).click();
  await plant.click();
  await expect(mass).toHaveValue('1');
  const write = page.waitForResponse((response) => response.request().method() === 'PATCH'
    && new URL(response.url()).pathname === projectPath);
  await enter(mass, '2');
  const saved = await write;
  expect(saved.ok(), await saved.text()).toBe(true);
  expect(saved.request().postDataJSON()).toEqual({
    revision: before.revision, address: { kind: 'variant', id: 'compare' },
    patches: [{ op: 'add', path: '/patches', value: { 'documents.model.payload.nodes.plant.params.mass': { value: 2 } } }],
  });
  const edited = await saved.json() as StudioProjectSnapshot;
  expect(edited.project.documents).toEqual(before.project.documents);
  expect(edited.project.steps).toEqual(before.project.steps);
  expect(edited.file_revisions.filter(({ address }) => !(address.kind === 'variant' && address.id === 'compare')))
    .toEqual(before.file_revisions.filter(({ address }) => !(address.kind === 'variant' && address.id === 'compare')));
  expect(edited.project.sweep).toEqual({
    ...before.project.sweep, rows: {
      ...before.project.sweep?.rows, compare: {
        ...before.project.sweep?.rows?.compare,
        patches: { 'documents.model.payload.nodes.plant.params.mass': { value: 2 } },
      },
    },
  });
  const changed = await rows(edited.revision);
  for (const variant of ['base', 'control']) {
    expect(changed[variant].request_id).toBe(initial[variant].request_id);
    expect(changed[variant].result).toEqual(initial[variant].result);
  }
  expect(changed.compare.request_id).not.toBe(initial.compare.request_id);
  expect(changed.compare.result).toBeNull();

  async function command(name: 'Undo' | 'Redo') {
    await page.getByRole('button', { name: 'Command Palette', exact: true }).click();
    await page.getByRole('dialog', { name: 'Command Palette' }).getByRole('option', { name, exact: true }).click();
  }
  await expect(mass).toHaveValue('2');
  await command('Undo');
  await expect(mass).toHaveValue('1');
  const undone = await rows((await snapshot()).revision);
  expect(undone.compare.request_id).toBe(initial.compare.request_id);
  expect(undone.compare.result).toEqual(initial.compare.result);
  await command('Redo');
  await expect(mass).toHaveValue('2');

  await page.reload();
  await expect(variants).toHaveValue('base');
  await variants.selectOption('compare');
  await mapStep(page, 'train').click();
  await page.getByRole('navigation', { name: 'Details', exact: true }).getByRole('tab', { name: 'Model', exact: true }).click();
  await plant.click();
  await expect(mass).toHaveValue('2');
  expect((await snapshot()).revision).toBe(edited.revision);
});
