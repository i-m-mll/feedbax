import { expect, test } from './fixtures/studio';
import type { APIRequestContext, Page } from '@playwright/test';
import type { StudioLoadedModelImportResult, StudioProjectWindowMapProblem, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { appendManagedFiles, fixture, mapStep, runProjectFixture, seedManagedProject } from './fixtures/canonicalTrain';

test.use({ viewport: { width: 1440, height: 900 } });

/** The Problems the Service reports, one per place and reason whatever the number of runs it holds for. */
function serviceProblems(problems: readonly StudioProjectWindowMapProblem[]) {
  return new Set(problems.filter(({ code }) => code !== 'blocked_input').map((problem) => JSON.stringify(problem.package != null
    ? [problem.package] : [problem.step, problem.variant, problem.place, problem.path, problem.detail]))).size;
}

const variables = (page: Page) => page.locator('[data-slot="primary"]').getByRole('table', { name: 'Variables', exact: true });
const variableRow = (page: Page, name: string) => variables(page).getByRole('row')
  .filter({ has: page.getByRole('textbox', { name: `Rename ${name}`, exact: true }) });

/** What the Task view of the Evaluate that lists its own Set shows of it: its entries and the family's variable. */
async function listedSet(page: Page) {
  const entries = variables(page).getByRole('row').filter({ has: page.getByRole('list', { name: 'Values Command kicks', exact: true }) })
    .getByRole('listitem');
  await expect.soft(entries, 'The listed Set shows its two entries').toHaveCount(2);
  await expect.soft(entries.filter({ hasText: /^Command pulse \{axis\} \{sign\}\s*× 4/ }), 'The family is one entry with its count').toHaveCount(1);
  const kick = variableRow(page, 'Kick');
  await expect.soft(kick, 'The family\'s variable is one row').toHaveCount(1);
  await expect.soft(kick.getByRole('img', { name: /^Set Entry Command pulse \{axis\} \{sign\}/ }), 'The variable is marked with its entry').toHaveCount(1);
  const header = page.locator('[data-slot="primary"] .studio-window-slot__header');
  await expect.soft(header.locator('button[data-document] [data-status-mark]'), 'The Task in use has no Problem mark').toHaveCount(0);
}

/** What the Task view of the Evaluate that inherits its subject's Set shows: each of the Set's variables once. */
async function inheritedSet(page: Page) {
  for (const name of ['Kick', 'Bias']) await expect.soft(variableRow(page, name), `${name} is one row`).toHaveCount(1);
}

const module = 'web.e2e.studio.fixtures.canonical_train';

/** The hold campaign's project, with its Loaded Model loaded through the Service and the Evaluate step that reads it. */
async function holdCampaign(request: APIRequestContext) {
  const { snapshot: seeded, projectPath } = await seedManagedProject(request, await fixture('hold-campaign-files'));
  const archive = Buffer.from(JSON.parse(await runProjectFixture(module, 'hold-campaign-archive')), 'base64');
  const loaded = await request.post(`${projectPath}/loaded-models/import`, {
    params: { revision: seeded.revision, name: 'loaded-controllers', model: 'campaign-point-mass', label: 'Loaded controllers' },
    headers: { 'content-type': 'application/octet-stream' }, data: archive,
  });
  expect(loaded.status(), await loaded.text()).toBe(201);
  const result = await loaded.json() as StudioLoadedModelImportResult;
  if (result.status !== 'imported') throw new Error(`The archive did not load: ${JSON.stringify(result)}`);
  return appendManagedFiles(request, result.snapshot, await fixture('loaded-check-files'));
}

// Catches a view that fails to render a Set entry family, a lowered Set member, a trial family, an Objective family, a
// based Model or a Loaded Model, a Problems count that differs from the Service's, and a member key or generated id shown as a name.
test('every view of every step of a hold-campaign project renders, with only Service names', { tag: '@round' }, async ({ page, request, memberLabels }) => {
  const { snapshot, projectPath } = await holdCampaign(request);
  const plan = await request.get(`${projectPath}/train-runs`, { params: { revision: snapshot.revision, window_map: 'true' } });
  expect(plan.ok(), await plan.text()).toBe(true);
  const expected = serviceProblems((await plan.json() as StudioSavedTrainRuns).window_map!.problems);
  const tabs = page.getByRole('navigation', { name: 'Details', exact: true }).locator('[role="tab"]:not([aria-disabled="true"])');
  const failures = page.getByRole('alert');
  const problems = page.getByRole('button', { name: /^Problems \d+$/ });

  const setViews = new Set<string>();
  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  for (const step of Object.keys(snapshot.project.steps ?? {})) {
    await mapStep(page, step).click();
    await expect(tabs.first()).toBeVisible();
    await expect(page.locator('[aria-busy="true"]')).toHaveCount(0);
    for (const view of await tabs.evaluateAll((items) => items.map((item) => item.dataset.commandId!))) {
      await test.step(`${step} ${view}`, async () => {
        const tab = tabs.and(page.locator(`[data-command-id="${view}"]`));
        await tab.click();
        await expect(tab).toHaveAttribute('aria-selected', 'true');
        await expect(page.locator('[aria-busy="true"]')).toHaveCount(0);
        expect.soft(await failures.allInnerTexts(), `${step} ${view} shows no failure`).toEqual([]);
        await expect.soft(problems, 'Studio counts the Service\'s Problems').toHaveAccessibleName(`Problems ${expected}`);
        const sets = view === 'task' && { 'loaded-check': listedSet, 'kick-check': inheritedSet }[step];
        if (sets) {
          await sets(page);
          setViews.add(step);
        }
        await memberLabels(`${step} ${view}`);
      });
    }
  }
  expect(setViews, 'Both Evaluates\' Task views were checked').toEqual(new Set(['loaded-check', 'kick-check']));
});
