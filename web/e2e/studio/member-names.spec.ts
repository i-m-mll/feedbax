import { expect, test } from './fixtures/studio';
import type { Locator } from '@playwright/test';
import type { StudioJobSubmission, StudioSavedTrainRuns } from '../../src/generated/studioContracts';
import { fixture, mapStep, seedManagedProject } from './fixtures/canonicalTrain';
import { activityJob, documentPicker, problemGroup } from './fixtures/members';

test.use({ viewport: { width: 1440, height: 900 } });

// The one spec whose claim is naming: a member reads the same wherever Studio shows it, an unlabelled member never reads
// as its key, and a labelled member reads as its label. Every expected value comes from the fixture; the spec computes
// no name.

const text = async (locator: Locator) => (await locator.innerText()).trim();

test('a member reads the same in the map, Inspector, picker, Problems and Activity', { tag: '@round' }, async ({ page, request }) => {
  test.setTimeout(60_000);
  const [model, task, objective, train, sibling] = await fixture('history-files');
  delete task.document.label;
  const siblingLabel = 'Reach sibling';
  sibling.step.label = siblingLabel;
  const broken = { kind: 'step', id: 'broken_reach', step: { kind: 'train', base: { inline: {
    trial_documents: train.step.base.inline.trial_documents,
  } } } };
  const { snapshot, projectPath, configuration } = await seedManagedProject(request, [model, task, objective, train, sibling, broken]);
  const runsPath = `${projectPath}/train-runs`;
  const shown = new Map<string, string[]>();
  const read = async (member: string, locator: Locator) => {
    await expect(locator).toBeVisible();
    shown.set(member, [...shown.get(member) ?? [], await text(locator)]);
  };
  const jobs = new Map<string, string>();
  async function run(step: string) {
    await mapStep(page, step).click();
    await page.getByRole('table', { name: 'Runs' }).getByRole('button').first().click();
    const admission = page.waitForResponse((response) =>
      response.request().method() === 'POST' && new URL(response.url()).pathname === runsPath);
    await page.getByRole('button', { name: 'Run', exact: true }).click();
    const response = await admission;
    expect(response.status(), await response.text()).toBe(202);
    const [item] = (await response.json() as StudioJobSubmission).items;
    expect(item?.disposition).toBe('admitted');
    jobs.set(step, item!.job_id!);
    await expect.poll(async () => {
      const { revision } = await (await request.get(projectPath)).json() as { revision: string };
      const rows = await (await request.get(`${runsPath}?window_map=true&revision=${revision}`)).json() as StudioSavedTrainRuns;
      return rows.rows.find(({ key }) => key.step === step)?.status;
    }, { timeout: 20_000 }).toBe('succeeded');
  }
  const inspector = page.getByRole('complementary', { name: 'Inspector', exact: true }).getByRole('heading', { level: 3 });

  await page.goto(`/?project=${encodeURIComponent(snapshot.project_id)}`);
  for (const step of ['train', 'sibling', 'broken_reach']) {
    await read(`step:${step}`, page.locator(`[data-map-step="${step}"] .studio-map-card__head`));
    await mapStep(page, step).click();
    await read(`step:${step}`, inspector);
  }
  await run('train');
  await run('sibling');

  await mapStep(page, 'train').click();
  const header = page.locator('[data-slot="primary"]').getByRole('tabpanel').locator('.studio-window-slot__header');
  await page.getByRole('tab', { name: 'Model', exact: true }).click();
  await read(`document:${model.id}`, documentPicker(header, model.id));
  await page.getByRole('tab', { name: 'Task', exact: true }).click();
  await read(`document:${task.id}`, documentPicker(header, task.id));

  await page.getByRole('button', { name: /^Problems \d+$/ }).click();
  await read('step:broken_reach', problemGroup(page.getByRole('dialog', { name: 'Problems' }), 'step:broken_reach').locator('h2'));
  await page.keyboard.press('Escape');

  await page.getByRole('button', { name: /^Activity( \d+)?$/ }).click();
  const activity = page.getByRole('region', { name: 'Activity', exact: true }).getByRole('region', { name: configuration.project, exact: true });
  for (const step of ['train', 'sibling']) await read(`step:${step}`, activityJob(activity, jobs.get(step)!).locator('button > span:last-child'));

  for (const [member, names] of shown) expect(new Set(names), member).toHaveProperty('size', 1);
  for (const member of ['step:train', 'step:broken_reach', `document:${task.id}`]) {
    expect(shown.get(member)![0], member).not.toBe(member.slice(member.indexOf(':') + 1));
  }
  expect(shown.get('step:sibling')![0]).toBe(siblingLabel);
  expect(shown.get(`document:${model.id}`)![0]).toBe(model.document.label);
});
