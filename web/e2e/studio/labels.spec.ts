import { expect, test } from './fixtures/studio';
import type { Page } from '@playwright/test';
import type {
  AnalysisDocument, StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import { mapStep, runProjectFixture } from './fixtures/canonicalTrain';
import { rootPayload } from './fixtures/rootPayload';

test.use({ viewport: { width: 1440, height: 900 } });

const tab = (page: Page, name: string) => page.getByRole('tab', { name, exact: true });

test('Rename labels a step, an operation and a figure; Change Key rewrites the Report and dims a held key', async ({ page, request, studio }) => {
  test.setTimeout(60_000);
  const module = 'web.e2e.studio.fixtures.managed_report';
  const project = JSON.parse(await runProjectFixture(module, 'seed_labels', studio.home, 'labels'));
  const path = '/api/projects/' + project;
  const analysis = page.getByRole('tabpanel', { name: 'Analysis', exact: true });
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true });
  const graph = (saved: StudioProjectSnapshot) =>
    rootPayload(saved.project.documents?.['velocity-analysis']) as unknown as AnalysisDocument;

  async function snapshot() {
    const response = await request.get(path + '/open');
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectDocument).snapshot;
  }
  async function requests(revision: string) {
    return JSON.parse(await runProjectFixture(module, 'requests', studio.home, project, revision));
  }
  async function reportProblems(revision: string) {
    const response = await request.get(path + '/train-runs', { params: { revision, window_map: 'true' } });
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioSavedTrainRuns).window_map!.problems.filter(({ step }) => step === 'report');
  }
  /** Waits for the member command `action` sends and returns the project it leaves. */
  async function member(action: () => Promise<unknown>) {
    const sent = page.waitForResponse((response) => response.request().method() === 'POST' &&
      new URL(response.url()).pathname === path + '/members/command');
    await action();
    const response = await sent;
    expect(response.ok(), await response.text()).toBe(true);
    return await snapshot();
  }
  async function canvasMenu(name: string, command: string) {
    await analysis.getByRole('group', { name: `Select ${name}`, exact: true }).click({ button: 'right' });
    await page.getByRole('menuitem', { name: new RegExp(`^${command}\\b`) }).click();
  }

  await page.goto('/?project=' + encodeURIComponent(project));
  const before = await snapshot();
  const identities = await requests(before.revision);
  expect(await reportProblems(before.revision)).toEqual([]);

  await mapStep(page, 'analyze').click();
  await mapStep(page, 'analyze').press('F2');
  const stepName = page.locator('[data-map-step="analyze"]').getByRole('textbox', { name: 'Rename Analyze', exact: true });
  const labelled = await member(() => stepName.fill('Velocity').then(() => stepName.press('Enter')));
  expect(labelled.project.steps?.analyze.label).toBe('Velocity');
  expect(Object.keys(labelled.project.steps ?? {}).sort()).toEqual(Object.keys(before.project.steps ?? {}).sort());
  expect(labelled.project.steps?.report.inputs).toEqual(before.project.steps?.report.inputs);
  expect(await requests(labelled.revision)).toEqual(identities);
  await expect(page.locator('[data-map-step="analyze"]').getByText('Velocity', { exact: true })).toBeVisible();

  await mapStep(page, 'analyze').dblclick();
  await tab(page, 'Analysis').click();
  await canvasMenu('Summarize', 'Rename');
  const operationName = analysis.getByRole('textbox', { name: 'Rename Summarize', exact: true });
  await expect(operationName).toBeFocused();
  const operation = await member(() => operationName.fill('Summary').then(() => operationName.press('Enter')));
  expect(graph(operation).operations?.summarize.label).toBe('Summary');
  expect(await requests(operation.revision)).toEqual(identities);
  expect(operation.project.steps?.report.inputs).toEqual(before.project.steps?.report.inputs);
  expect(await reportProblems(operation.revision)).toEqual([]);
  await expect(analysis.getByRole('group', { name: 'Select Summary', exact: true })).toBeVisible();

  await canvasMenu('Summary', 'Change Key');
  const operationKey = inspector.getByRole('textbox', { name: 'Change Key Summary', exact: true });
  await expect(operationKey).toBeFocused();
  await operationKey.fill('held');
  await expect(operationKey).toHaveAttribute('aria-invalid', 'true');
  await operationKey.press('Enter');
  await expect(operationKey).toBeFocused();
  await operationKey.press('Escape');
  expect((await snapshot()).revision).toBe(operation.revision);

  await canvasMenu('Summary', 'Change Key');
  const rekeyed = await member(() => operationKey.fill('summary').then(() => operationKey.press('Enter')));
  expect(Object.keys(graph(rekeyed).operations ?? {}).sort()).toEqual(['align', 'summary']);
  expect(graph(rekeyed).views?.profile.inputs.summary).toEqual({ element: 'summary', product: 'summary' });
  expect(await reportProblems(rekeyed.revision)).toEqual([]);

  await canvasMenu('Profile', 'Change Key');
  const figureKey = inspector.getByRole('textbox', { name: 'Change Key Profile', exact: true });
  const figure = await member(() => figureKey.fill('velocity-profile').then(() => figureKey.press('Enter')));
  expect(figure.project.steps?.report.inputs?.velocity_summary).toMatchObject({ step: 'analyze', element: 'velocity-profile' });
  expect(await reportProblems(figure.revision)).toEqual([]);
});
