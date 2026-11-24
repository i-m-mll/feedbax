import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { ModelDocument, StudioProjectSnapshot } from '../../src/generated/studioContracts';
import { fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';
import { readSaved } from './fixtures/studioKeyboard';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const inspector = (page: Page) => page.locator('[data-studio-focus-region="inspector"]:visible');

function components(saved: StudioProjectSnapshot, document: string) {
  return Object.entries((rootPayload(saved.project.documents?.[document]) as unknown as ModelDocument).interior.components ?? {});
}

test('the Model inspector frames the document, one component and two', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('managed-files'));
  await page.goto(`/?project=${snapshot.project_id}`);
  await mapStep(page, 'train').press('Enter');
  await page.keyboard.press('Meta+2');
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  const panel = inspector(page);
  const title = panel.locator('header').first();

  await expect(title).toHaveText('point-massModel');
  await expect(panel.getByRole('textbox', { name: 'Step', exact: true })).toBeVisible();

  const card = (name: string) => model.getByRole('article', { name, exact: true });
  await card('Motor command').locator('header').click();
  await expect(title).toHaveText('Motor commandLinear');
  await expect(panel.locator('strong')).toHaveCount(0);
  const [id] = components(snapshot, 'point-mass').find(([, item]) => item.name === 'Motor command')!;
  await expect(panel).not.toContainText(id);
  await expect(panel).not.toContainText('output');
  const add = panel.getByRole('button', { name: 'Add Attachment', exact: true });
  await expect(add).toBeVisible();
  await expect(panel.locator('.border-t')).toHaveCount(1);

  const [, original] = components(snapshot, 'point-mass').find(([key]) => key === id)!;
  const before = JSON.stringify(original.kind === 'leaf' ? original.fields : null);
  await page.keyboard.press('Meta+D');
  let copyId = '';
  await expect.poll(async () => {
    const motors = components(await readSaved(request, projectPath), 'point-mass').filter(([, item]) => item.name === 'Motor command');
    copyId = motors.find(([key]) => key !== id)?.[0] ?? '';
    return motors.length;
  }).toBe(2);
  const node = (key: string) => model.locator(`.react-flow__node[data-id="${key}"] article`);
  await expect(node(copyId)).toHaveAttribute('data-selected');
  await expect(node(id)).not.toHaveAttribute('data-selected');
  const field = panel.getByRole('textbox', { name: /^bias init$/i });
  await field.fill('0.5');
  await field.press('Enter');
  await expect(field).toHaveValue('0.5');

  await node(id).locator('header').click({ modifiers: ['Shift'] });
  await expect(title).toHaveText('2 components');
  const shared = panel.getByRole('textbox', { name: /^bias init$/i });
  await expect(shared).toHaveValue('');
  await expect(shared).toHaveAttribute('placeholder', 'Mixed');
  await shared.fill('0.25');
  await shared.press('Enter');
  await expect(shared).toHaveValue('0.25');
  await expect(node(copyId)).toHaveAttribute('data-selected');
  await expect(node(id)).toHaveAttribute('data-selected');
  await expect.poll(async () => {
    const saved = Object.fromEntries(components(await readSaved(request, projectPath), 'point-mass'));
    const [original, duplicate] = [saved[id], saved[copyId]].map((item) => JSON.stringify(item.kind === 'leaf' ? item.fields : null));
    return original === duplicate && original !== before;
  }).toBe(true);
});

test('each Layers entry hides one class, and selecting a body in the Scene selects its card in the Model', async ({ page, studio }) => {
  const id = 'inspector-scene';
  const folder = join(studio.home, 'state', 'web', 'projects', id);
  await runProjectFixture('tests.fixtures.scene_geometry', 'seed', folder);
  await mkdir(join(folder, 'steps'), { recursive: true });
  await writeFile(join(folder, 'steps', 'train.json'), JSON.stringify({
    kind: 'train', base: { inline: { trial_documents: { model: { document: 'chain' } } } },
  }));
  await page.goto(`/?project=${id}`);
  await mapStep(page, 'train').press('Enter');
  await page.keyboard.press('Meta+2');
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await model.getByRole('article', { name: 'Mechanics', exact: true }).dblclick();
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const scene = page.getByRole('tabpanel', { name: 'Scene', exact: true });
  const upper = scene.getByRole('button', { name: 'Select upper', exact: true }).filter({ has: page.locator('polyline') });
  const drawn = { Bodies: upper, Joints: scene.getByRole('button', { name: 'Select shoulder', exact: true }),
    Muscles: scene.getByRole('button', { name: 'Select cross_chain', exact: true }) };
  for (const [layer, hidden] of Object.entries(drawn)) {
    const entry = page.getByRole('menuitemcheckbox', { name: layer, exact: true });
    await scene.getByRole('button', { name: 'Layers', exact: true }).click();
    await entry.click();
    await expect(hidden).toHaveCount(0);
    for (const other of Object.values(drawn)) if (other !== hidden) await expect(other).toBeVisible();
    await entry.click();
    await expect(hidden).toBeVisible();
    await page.keyboard.press('Escape');
  }
  await upper.press('Enter');
  await expect(upper).toHaveAttribute('aria-pressed', 'true');
  await expect(model.locator('.react-flow__node[data-id="upper"] article')).toHaveAttribute('data-selected');
});
