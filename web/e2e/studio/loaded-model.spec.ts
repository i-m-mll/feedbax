import { expect, test } from './fixtures/studio';
import { loadedModelDocumentSchemaId, type StudioLoadedModelImportResult, type StudioProjectDocument } from '../../src/generated/studioContracts';
import { runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

const digest = (character: string) => character.repeat(64);

// Catches a model file that does not fit refused with only its first problem, or away from the Model it was loaded
// against; a model file that fits imported but not listed under its Model, or not shown once loaded; and a loaded
// model that still offers the Library or a field's kind menu.
test('Load reports every problem of a model file at the Model picker, and shows one that fits under its Model', async ({ page, request }) => {
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.evaluate_authoring'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const imports: URL[] = [];
  let fits = false;
  await page.route(`**${projectPath}/loaded-models/import?*`, async (route) => {
    const url = new URL(route.request().url());
    imports.push(url);
    if (!fits) {
      const result: StudioLoadedModelImportResult = { status: 'does_not_fit', report: { problems: [
        { kind: 'missing_slot', replica: '0', address: 'controller.weight_hh', detail: 'absent' },
        { kind: 'unexpected_slot', replica: '0', address: 'controller.extra', detail: 'not in the Model' },
        { kind: 'mismatched_slot', replica: '1', address: 'readout.weight', detail: 'shape (2, 8) ≠ (2, 4)' },
      ] } };
      return route.fulfill({ status: 422, json: result });
    }
    const name = url.searchParams.get('name')!;
    const created = await request.post(`${projectPath}/files`, { data: { revision: url.searchParams.get('revision'), file: {
      kind: 'document', id: name, document: { schema: loadedModelDocumentSchemaId, payload: {
        model: { document: url.searchParams.get('model') },
        model_revision: { sha256: digest('a'), size_bytes: 1200 }, weights: { sha256: digest('b'), size_bytes: 800 },
        encoding_sha256: digest('c'),
      } },
    } } });
    expect(created.status(), await created.text()).toBe(201);
    const result: StudioLoadedModelImportResult = {
      status: 'imported', document: name, snapshot: (await created.json() as StudioProjectDocument).snapshot,
    };
    return route.fulfill({ status: 201, json: result });
  });

  await page.goto(`/?project=${snapshot.project_id}`);
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  const picker = page.getByRole('button', { name: /^Model Document / });
  const load = async () => {
    await picker.click();
    const chooser = page.waitForEvent('filechooser');
    await page.getByRole('menuitem', { name: 'Load', exact: true }).click();
    await (await chooser).setFiles({ name: 'reach baseline.feedbax', mimeType: 'application/octet-stream', buffer: Buffer.from('archive') });
  };

  await load();
  const misfit = page.getByRole('dialog', { name: 'Misfit' });
  await expect(misfit.getByRole('listitem')).toHaveText([
    'controller.weight_hh missing · 0', 'controller.extra unexpected · 0', 'readout.weight shape (2, 8) ≠ (2, 4) · 1',
  ]);
  expect(imports[0].searchParams.get('model')).toBe('point-mass');
  await page.keyboard.press('Escape');
  await expect(misfit).toBeHidden();

  fits = true;
  await load();
  await expect(page.getByRole('button', { name: 'Loaded Model Reach baseline' })).toBeVisible();
  const loaded = page.getByRole('complementary', { name: 'Inspector' });
  await expect(loaded.getByRole('region', { name: 'Reach baseline', exact: true }).getByText('Point mass', { exact: true })).toBeVisible();
  await expect(loaded.getByRole('region', { name: 'Point mass', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Loaded Model Reach baseline' }).click();
  const items = page.getByRole('menu', { name: 'Loaded Model' }).getByRole('menuitemradio');
  await expect(items).toHaveText(['Point mass', 'Reach baseline']);
  await page.keyboard.press('Escape');

  // A loaded model is read-only: no Library to place from, and a component's kind marks open nothing.
  await expect(page.getByRole('textbox', { name: 'Search Library', exact: true })).toHaveCount(0);
  await page.getByRole('treeitem', { name: 'Recurrent controller', exact: true }).click();
  const inspector = page.getByRole('complementary', { name: 'Inspector' });
  await expect(inspector.getByRole('img', { name: /kind: Value$/ }).first()).toBeVisible();
  await expect(inspector.getByRole('button', { name: /kind: / })).toHaveCount(0);
});
