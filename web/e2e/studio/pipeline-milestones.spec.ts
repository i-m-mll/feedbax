import { expect, test, type APIRequestContext, type Page } from '@playwright/test';
import type {
  StudioActivityJob, StudioJobEventHistory, StudioJobSnapshot, StudioJobSubmission,
  StudioProjectDocument, StudioProjectSnapshot, StudioSavedTrainKey, StudioSavedTrainRuns,
} from '../../src/generated/studioContracts';
import {
  appendManagedFiles, authorLearningRateSweep, enter, fixture, mapStep, openResultsSlot,
  runProjectFixture, seedManagedProject,
} from './fixtures/canonicalTrain';

const primary = (page: Page) => page.locator('[data-studio-focus-region="primary"]');
const runs = (page: Page) => primary(page).getByRole('table', { name: 'Runs', exact: true });
const runButton = (page: Page) => primary(page).getByRole('button', { name: 'Run', exact: true });

function responseFor(page: Page, path: string) {
  return page.waitForResponse((response) =>
    response.request().method() === 'POST' && new URL(response.url()).pathname === path,
  );
}

async function select(page: Page, step: string) {
  await mapStep(page, step).click();
  await primary(page).focus();
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
}

async function admit(page: Page, projectPath: string) {
  const posted = responseFor(page, projectPath + '/train-runs');
  await runButton(page).click();
  const response = await posted;
  expect(response.status(), await response.text()).toBe(202);
  const { items } = await response.json() as StudioJobSubmission;
  expect(items).toHaveLength(1);
  expect(items[0].disposition).toBe('admitted');
  return items[0];
}

async function savedPlan(request: APIRequestContext, projectPath: string) {
  const definition = await request.get(projectPath);
  expect(definition.ok(), await definition.text()).toBe(true);
  const { revision } = await definition.json() as StudioProjectSnapshot;
  const response = await request.get(projectPath + '/train-runs', { params: { revision, window_map: 'true' } });
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioSavedTrainRuns;
}

async function launchRows(page: Page, projectPath: string, keys: StudioSavedTrainKey[]) {
  const posted = responseFor(page, projectPath + '/train-runs');
  await runButton(page).click();
  const response = await posted;
  expect(response.status(), await response.text()).toBe(202);
  expect(response.request().postDataJSON().run_keys).toEqual(keys);
  const { items } = await response.json() as StudioJobSubmission;
  expect(items.map(({ disposition }) => disposition)).toEqual(keys.map(() => 'admitted'));
}

test('M1 starts an ordinary run, watches loss, and resumes the checkpoint from Activity',
  async ({ page, request }) => {
    test.setTimeout(50_000);
    let creates = 0;
    page.on('request', (request) => {
      if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/projects') creates += 1;
    });
    const creating = responseFor(page, '/api/projects');
    await page.goto('/');
    const created = await creating;
    expect(created.status()).toBe(201);
    const { snapshot } = await created.json() as StudioProjectDocument;
    const id = snapshot.project_id;
    await expect(page).toHaveURL(new RegExp('project=' + id + '$'));
    const { projectPath } = await appendManagedFiles(request, snapshot, await fixture('ordinary-files', '1200'));
    await page.reload();
    await select(page, 'train');
    expect(creates).toBe(1);
    const item = await admit(page, projectPath);
    const jobId = item.job_id!;
    await runs(page).getByRole('button').first().click();
    await openResultsSlot(page);
    const plot = page.getByRole('img', { name: 'Loss', exact: true });
    await expect(plot.locator('line title').filter({ hasText: /samples?/ }).first()).toBeAttached({ timeout: 14_000 });

    await page.getByRole('button', { name: 'Activity', exact: true }).click();
    const activity = page.getByRole('region', { name: 'Activity', exact: true });
    const attempt = activity.getByRole('list', { name: 'Runs' }).getByRole('listitem').filter({ hasText: id });
    const stopping = responseFor(page, '/api/jobs/stop');
    await attempt.getByRole('button', { name: 'Stop', exact: true }).click();
    expect((await stopping).status()).toBe(202);
    const resume = attempt.getByRole('button', { name: 'Resume', exact: true });
    await expect(resume).toBeEnabled({ timeout: 10_000 });
    const stopped = await request.get('/api/jobs', { params: { project: id } });
    expect(stopped.ok()).toBe(true);
    const snapshotOfJobs = await stopped.json() as StudioJobSnapshot;
    const parent = snapshotOfJobs.jobs.find((job) => job.job_id === jobId);
    expect(parent).toMatchObject({ status: 'stopped', can_resume: true });

    const resuming = responseFor(page, `/api/jobs/${jobId}/resume`);
    await resume.click();
    const resumed = await resuming;
    expect(resumed.status(), await resumed.text()).toBe(202);
    const continuation = await resumed.json() as StudioActivityJob;
    expect(continuation).toMatchObject({
      continues: jobId, request_id: parent!.request_id, run_key: item.run_key,
    });
    expect(continuation.job_id).not.toBe(jobId);

    const replayed = page.waitForResponse((response) => {
      const url = new URL(response.url());
      return url.pathname === '/api/events/history' && url.searchParams.get('project') === id;
    });
    await page.reload();
    await select(page, 'train');
    const replay = await replayed;
    expect(replay.ok(), await replay.text()).toBe(true);
    const history = await replay.json() as StudioJobEventHistory;
    expect(history.events.some((event) => event.job_id === jobId && event.kind === 'progress')).toBe(true);
    await expect(runs(page).getByText('Done', { exact: true })).toBeVisible({ timeout: 16_000 });
  },
);

test('Activity shows the original CLI and Studio clients for a saved Train request', async ({ page, request }, info) => {
  test.setTimeout(45_000);
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '2'));
  const project = snapshot.project_id;
  const label = 'Batch review';
  await page.goto('/?project=' + project);
  await select(page, 'train');
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const activity = page.getByRole('region', { name: 'Activity', exact: true });
  const entries = activity.getByRole('list', { name: 'Runs', exact: true }).getByRole('listitem').filter({ hasText: project });

  await runProjectFixture('web.e2e.studio.fixtures.canonical_train',
    'run-saved-cli', String(info.config.metadata.home), project, label);
  const cli = entries.filter({ has: page.getByText('CLI', { exact: true }) });
  await expect(cli.getByText(label, { exact: true })).toBeVisible();
  await expect(cli.getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });
  await expect(runs(page).getByText('Done', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const studioJob = await admit(page, projectPath);
  await page.getByRole('button', { name: 'Activity', exact: true }).click();
  const studio = entries.filter({ has: page.getByText('Studio', { exact: true }) });
  await expect(studio.getByText('Client', { exact: true })).toBeVisible();
  await expect(studio.getByText('Label', { exact: true })).toHaveCount(0);
  await expect(cli.getByText(label, { exact: true })).toBeVisible();
  await expect(studio.getByText('Done', { exact: true })).toBeVisible({ timeout: 18_000 });

  const response = await request.get('/api/jobs', { params: { project } });
  expect(response.ok(), await response.text()).toBe(true);
  const { jobs } = await response.json() as StudioJobSnapshot;
  expect(jobs).toHaveLength(2);
  expect(jobs.find(({ job_id }) => job_id === studioJob.job_id)?.actor).toEqual({ client: 'studio' });
  expect(jobs.find(({ actor }) => actor?.client === 'cli')?.actor).toEqual({ client: 'cli', label });
  expect(new Set(jobs.map(({ request_id }) => request_id)).size).toBe(1);
  await info.attach('activity-actors', {
    body: JSON.stringify(jobs.map(({ job_id, request_id, actor }) => ({ job_id, request_id, actor }))),
    contentType: 'application/json',
  });
});

test('M1 runs only the selected row of a sweep authored in Studio', async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '30'));
  await page.goto('/?project=' + snapshot.project_id);
  await mapStep(page, 'train').click();
  const rate = page.getByRole('region', { name: 'Learning Rate field', exact: true });
  await authorLearningRateSweep(rate);
  await page.getByRole('tab', { name: 'Runs', exact: true }).click();
  const picked = runs(page).getByRole('button', { name: /Medium/ });
  await picked.click();
  const submitted = await admit(page, projectPath);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(1);
  await expect(runs(page).getByText('Not run', { exact: true })).toHaveCount(2);
  const response = await request.get('/api/jobs', { params: { project: snapshot.project_id } });
  expect(response.ok()).toBe(true);
  const { jobs, cursor } = await response.json() as StudioJobSnapshot;
  expect(jobs).toHaveLength(1);
  expect(jobs[0]).toMatchObject({ run_key: submitted.run_key, job_id: submitted.job_id });
  const historyResponse = await request.get('/api/events/history', {
    params: { project: snapshot.project_id, through: cursor },
  });
  expect(historyResponse.ok()).toBe(true);
  const { events } = await historyResponse.json() as StudioJobEventHistory;
  const intervals = events
    .filter((event) => event.kind === 'progress' && event.job_id === jobs[0].job_id)
    .flatMap((event) => event.data.metric_interval ? [event.data.metric_interval] : []) as Array<{
      start_exclusive: number;
      end_inclusive: number;
      metrics: { train_loss: { count: number; min: number; mean: number; max: number } };
    }>;
  let end = 0;
  for (const interval of intervals) {
    const loss = interval.metrics.train_loss;
    expect(interval.start_exclusive).toBe(end);
    expect(loss.count).toBe(interval.end_inclusive - end);
    expect(loss.mean).toBeGreaterThanOrEqual(loss.min);
    expect(loss.mean).toBeLessThanOrEqual(loss.max);
    end = interval.end_inclusive;
  }
  expect(end).toBe(30);
});

for (const grouped of [false, true]) test(`M2 saves, runs, and freezes a ${grouped ? 'grouped' : 'measured'} Evaluate subset in Runs`, async ({ page, request }, info) => {
  test.setTimeout(59_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  const evaluation = 'web.e2e.studio.fixtures.canonical_evaluate';
  const metricName = 'program_step';
  const files = JSON.parse(await runProjectFixture(evaluation, grouped ? 'grouped-subset-files' : 'measured-subset-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const path = '/?project=' + snapshot.project_id;
  const readPlan = () => savedPlan(request, projectPath);
  await page.goto(path);
  await select(page, 'train');
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const train = page.getByRole('article', { name: 'Train train', exact: true });
  const created = responseFor(page, projectPath + '/files');
  await train.getByRole('button', { name: 'New Step train', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step' }).getByRole('button', { name: 'Evaluate', exact: true }).click();
  expect((await created).status()).toBe(201);
  const subset = drawer.getByRole('group', { name: 'Subject Subset', exact: true });
  const metric = subset.getByRole('textbox', { name: 'Metric', exact: true });
  const count = subset.getByRole('spinbutton', { name: 'Count', exact: true });
  const best = subset.getByRole('checkbox', { name: 'Best', exact: true });
  const apply = subset.getByRole('button', { name: 'Apply Subset', exact: true });
  const order = subset.getByRole('combobox', { name: 'Order', exact: true });
  const per = subset.getByRole('group', { name: 'Per', exact: true })
    .getByRole('checkbox', { name: 'replicate', exact: true });
  const inputs = drawer.getByRole('region', { name: 'Inputs', exact: true });
  const members = inputs.getByRole('list', { name: 'Members', exact: true }).getByRole('listitem');
  const groupLabels = ['replicate First', 'replicate Second'];
  const groupedLists = groupLabels.map((label) =>
    inputs.getByRole('list', { name: `Members ${label}`, exact: true }).getByRole('listitem'));
  const freeze = drawer.getByRole('button', { name: 'To Rows', exact: true });
  const saveWhere = async (click: () => Promise<void>) => {
    const saved = page.waitForResponse((response) => response.request().method() === 'PATCH'
      && new URL(response.url()).pathname === projectPath
      && JSON.stringify(response.request().postDataJSON()).includes('/inputs/subject/where'));
    await click();
    const response = await saved;
    expect(response.ok(), await response.text()).toBe(true);
    return (await response.json() as StudioProjectSnapshot).project.steps.evaluate.inputs?.subject;
  };

  await best.check();
  await expect(apply).toBeDisabled();
  await metric.fill(metricName);
  await count.fill('0');
  await expect(apply).toBeDisabled();
  await count.fill('2');
  if (grouped) await per.check();
  expect(await saveWhere(() => apply.click())).toMatchObject({
    step: 'train', checkpoint: 'last',
    where: { best: { by: metricName, k: 2, order: 'min', ...(grouped && { per: ['replicate'] }) } },
  });

  await page.reload();
  await select(page, 'evaluate');
  await expect(best).toBeChecked();
  await expect(metric).toHaveValue(metricName);
  await expect(count).toHaveValue('2');
  await expect(order).toHaveValue('min');
  await expect(inputs).toContainText(grouped ? '4 of 4' : '2 of 3');
  const memberLists = grouped ? groupedLists : [members];
  if (grouped) {
    await expect(per).toBeChecked();
    await expect(inputs).toContainText('Per replicate');
    await expect(inputs.getByRole('list')).toHaveCount(2);
    await expect(inputs.getByRole('list').nth(0)).toHaveAccessibleName(`Members ${groupLabels[0]}`);
    await expect(inputs.getByRole('list').nth(1)).toHaveAccessibleName(`Members ${groupLabels[1]}`);
  }
  for (const list of memberLists) await expect(list).toHaveText(['Rank 1', 'Rank 2']);
  const waiting = await readPlan();
  const ranks = waiting.window_map!.evaluations!.filter(({ key }) => key.step === 'evaluate');
  const projection = waiting.window_map!.steps.find(({ step_id }) => step_id === 'evaluate')!
    .inputs.find(({ id }) => id === 'subject')!.subset!;
  const keys = projection.ranks!;
  expect(ranks).toHaveLength(grouped ? 4 : 2);
  if (grouped) {
    expect(projection.rank_groups?.map(({ coordinates }) => coordinates)).toEqual([
      { replicate: 'zeta' }, { replicate: 'alpha' },
    ]);
    expect(projection.rank_groups?.map(({ ranks }) => ranks.length)).toEqual([2, 2]);
    expect(projection.rank_groups?.flatMap(({ ranks }) => ranks)).toEqual(keys);
    expect(ranks.map(({ subject_rank_key }) => subject_rank_key)).toEqual(keys);
    for (const [index, label] of groupLabels.entries()) {
      await expect(runs(page).getByRole('rowheader').nth(index * 2)).toHaveText(`${label} · Rank 1`);
      await expect(runs(page).getByRole('rowheader').nth(index * 2 + 1)).toHaveText(`${label} · Rank 2`);
    }
  } else {
    expect(keys).toEqual(['rank-1', 'rank-2']);
    for (const [index] of ranks.entries()) {
      await expect(runs(page).getByRole('row').filter({ hasText: new RegExp(`rank-${index + 1}`, 'i') })).toHaveCount(1);
    }
  }

  await select(page, 'train');
  const source = waiting.rows.filter(({ key }) => key.step === 'train');
  expect(source).toHaveLength(grouped ? 4 : 3);
  await launchRows(page, projectPath, source.map(({ key }) => key));
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(source.length, { timeout: 24_000 });
  const finishedPlan = await readPlan();
  const finished = finishedPlan.rows.filter(({ key }) => key.step === 'train');
  const attempts = finished.map(({ result }) => result!.job_id);
  const measures: Record<string, number> = JSON.parse(await runProjectFixture(
    evaluation, 'train-measures', String(info.config.metadata.home), snapshot.project_id, metricName, ...attempts,
  ));
  expect(attempts.map((attempt) => measures[attempt])).toEqual(grouped ? [4, 4, 2, 2] : [4, 3, 2]);
  const expected = grouped ? ['zeta', 'alpha'].flatMap((group) => {
    const candidates = finishedPlan.window_map!.rows.filter(({ key, coordinates }) =>
      key.step === 'train' && coordinates.replicate?.value_id === group);
    return candidates.toReversed().map(({ key }) => finished.find((row) => row.key.row === key.row)!);
  }) : [finished[2], finished[1]];

  await select(page, 'evaluate');
  for (const list of memberLists) await expect(list).toHaveText(['Rank 1', 'Rank 2']);
  await expect(freeze).toHaveCount(0);
  const ranked = (await readPlan()).window_map!.evaluations!.filter(({ key }) => key.step === 'evaluate');
  expect(ranked.map(({ key }) => key)).toEqual(ranks.map(({ key }) => key));
  if (grouped) {
    const firstGroup = runs(page).getByRole('rowheader').getByRole('button', { name: /^replicate First · Rank/ });
    await firstGroup.nth(0).click();
    await firstGroup.nth(1).click({ modifiers: ['Shift'] });
    await launchRows(page, projectPath, ranked.slice(0, 2).map(({ key }) => key));
    await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 18_000 });
    await select(page, 'evaluate');
    await expect(groupedLists[0]).toHaveText(expected.slice(0, 2).map(({ label }, index) => `Rank ${index + 1} · ${label}`));
    await expect(groupedLists[1]).toHaveText(['Rank 1', 'Rank 2']);
    await expect(freeze).toHaveCount(0);
  }
  await launchRows(page, projectPath, (grouped ? ranked.slice(2) : ranked).map(({ key }) => key));
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(ranked.length, { timeout: 18_000 });
  const completed = (await readPlan()).window_map!.evaluations!.filter(({ key }) => key.step === 'evaluate');
  for (const [index, producer] of expected.entries()) {
    expect(completed[index].bound_subject?.key).toBe(keys[index]);
    expect(completed[index].result?.checkpoint_source).toEqual({
      job_id: producer.result!.job_id, request_id: producer.request_id,
      completed_updates: measures[producer.result!.job_id],
    });
  }
  if (grouped) expect(new Set(completed.map(({ bound_subject }) => bound_subject!.selection_sha256)).size).toBe(2);
  for (const [index, producer] of expected.entries()) {
    const group = Math.floor(index / 2);
    const position = index % 2;
    await expect(memberLists[group].nth(position)).toHaveText(`Rank ${position + 1} · ${producer.label}`);
    await expect(runs(page).getByRole('row').filter({ hasText: producer.label })).toHaveCount(1);
    if (grouped) await expect(runs(page).getByRole('rowheader').nth(index))
      .toHaveText(`${groupLabels[group]} · Rank ${position + 1} · ${producer.label}`);
  }
  await expect(freeze).toBeEnabled();
  await count.fill('1');
  await expect(freeze).toBeDisabled();
  await count.fill('2');
  await expect(freeze).toBeEnabled();
  const preparing = responseFor(page, projectPath + '/authoring/subset-to-rows');
  const frozen = await saveWhere(() => freeze.click());
  const prepared = await preparing;
  expect(prepared.ok(), await prepared.text()).toBe(true);
  const preparedBody = prepared.request().postDataJSON();
  expect(preparedBody).toEqual({
    revision: expect.stringMatching(/^[a-f0-9]{64}$/), variant: 'default', step: 'evaluate',
    attempts: completed.map(({ result }) => result!.job_id),
  });
  expect(frozen).toMatchObject({
    step: 'train', checkpoint: 'last',
    where: { rows: expected.map(({ key: { variant, row } }, index) => ({ key: keys[index], variant, row })) },
  });
  expect(frozen).not.toHaveProperty('where.best');
  const staleFreeze = await request.post(projectPath + '/authoring/subset-to-rows', { data: preparedBody });
  expect(staleFreeze.status()).toBe(409);
  const replanned = (await readPlan()).window_map!.evaluations!.filter(({ key }) => key.step === 'evaluate');
  expect(replanned.map(({ key }) => key)).toEqual(completed.map(({ key }) => key));
  for (const [index, row] of replanned.entries()) expect(row.request_id).not.toBe(completed[index].request_id);
  await page.reload();
  await select(page, 'evaluate');
  await expect(best).not.toBeChecked();
  await expect(members).toHaveText(expected.map(({ label }) => label));
});

test('M2 authors and widens a saved Evaluate subset without resubmitting completed rows', async ({ page, request }) => {
  test.setTimeout(58_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  const evaluation = 'web.e2e.studio.fixtures.canonical_evaluate';
  const files = JSON.parse(await runProjectFixture(evaluation, 'walkthrough-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const mounted = '/?project=' + snapshot.project_id;
  const map = page.getByRole('region', { name: 'Pipeline Map', exact: true });
  const train = map.getByRole('article', { name: 'Train train', exact: true });
  const evaluated = map.getByRole('article', { name: 'Evaluate evaluate', exact: true });
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const steps = page.getByRole('complementary', { name: 'Map Drawer', exact: true });
  const readPlan = () => savedPlan(request, projectPath);

  await page.goto(mounted);
  await select(page, 'train');
  await authorLearningRateSweep(drawer.getByRole('region', { name: 'Learning Rate field', exact: true }));
  const sweep = train.getByRole('button', { name: /^Sweep learning-rate / });
  await expect(sweep).toHaveText('learning-rate 3');
  const definition = await readPlan();
  const sourceRows = definition.window_map!.rows.filter(({ key }) => key.step === 'train');
  expect(sourceRows.map(({ coordinates }) => coordinates['learning-rate'].label)).toEqual(['Slow', 'Medium', 'Fast']);
  expect(sourceRows.map(({ key, member_key }) => member_key === key.row)).toEqual([true, true, true]);
  const authoredRows = sourceRows.map(({ key: { variant, row }, member_key }) => ({ key: member_key, variant, row }));
  const ids = sourceRows.map(({ coordinates }) => coordinates['learning-rate'].value_id);
  const conditionIds = sourceRows[0].evaluation_conditions!.slice(0, 2);
  expect(conditionIds).toHaveLength(2);
  await launchRows(page, projectPath, sourceRows.map(({ key }) => key));
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(3, { timeout: 22_000 });

  await sweep.click();
  const value = (name: string) => train.getByRole('listitem').getByRole('button', {
    name: `Select learning-rate ${name}`, exact: true,
  });
  await value('Slow · 0.001').click();
  await value('Medium · 0.002').click({ modifiers: ['Shift'] });
  await expect(train).toContainText('2 of 3');
  const created = responseFor(page, projectPath + '/files');
  await train.getByRole('button', { name: 'New Step train', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step' }).getByRole('button', { name: 'Evaluate', exact: true }).click();
  const response = await created;
  expect(response.status(), await response.text()).toBe(201);
  const authored = (await response.json() as StudioProjectDocument).snapshot.project.steps.evaluate;
  expect(authored).toMatchObject({
    inputs: { subject: { step: 'train', checkpoint: 'last', where: { axes: { 'learning-rate': ids.slice(0, 2) } } } },
    sweep: { axes: { subject: { input: 'subject' } } },
  });
  const inputs = drawer.getByRole('region', { name: 'Inputs', exact: true });
  await expect(inputs).toContainText('learning-rate Slow · 0.001, Medium · 0.002 · 2 of 3');
  const members = inputs.getByRole('list', { name: 'Members', exact: true });
  await expect(members.getByRole('listitem')).toHaveText(sourceRows.slice(0, 2).map(({ label }) => label));
  const trialSet = drawer.getByRole('group', { name: 'Trial Set', exact: true });
  await expect(trialSet.getByRole('combobox')).toBeFocused();
  await trialSet.getByRole('combobox').selectOption('conditions');
  for (const condition of conditionIds) await trialSet.getByRole('checkbox', { name: condition, exact: true }).check();
  const edited = (path: string) => page.waitForResponse((response) => {
    const request = response.request();
    return request.method() === 'PATCH' && new URL(response.url()).pathname === projectPath &&
      JSON.stringify(request.postDataJSON()).includes(path);
  });
  const trialEdit = edited('/base/inline/params/trial_set');
  await trialSet.getByRole('button', { name: 'Apply Trials', exact: true }).click();
  const savedTrials = await trialEdit;
  expect(savedTrials.ok(), await savedTrials.text()).toBe(true);
  const step = (await savedTrials.json() as StudioProjectSnapshot).project.steps.evaluate;
  expect(step.base).toMatchObject({ inline: { params: { trial_set: { kind: 'conditions', ids: conditionIds } } } });
  expect(step.base).not.toHaveProperty('inline.params.trials');

  const keys = (await readPlan()).window_map!.evaluations!.map(({ key }) => key);
  expect(keys).toHaveLength(2);
  await launchRows(page, projectPath, keys);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(2, { timeout: 16_000 });
  const before = (await readPlan()).window_map!.evaluations!;
  await expect(runButton(page)).toBeDisabled();

  await select(page, 'train');
  const fast = runs(page).getByRole('row').filter({ hasText: 'Fast' }).getByRole('button');
  await fast.click();
  await expect(primary(page).getByRole('region', { name: 'Used By' })).toHaveCount(0);
  await expect(steps.getByRole('button', { name: 'Select evaluate Evaluate 0 of 2', exact: true })).toBeVisible();
  const slow = runs(page).getByRole('row').filter({ hasText: 'Slow' }).getByRole('button');
  await slow.click();
  await expect(primary(page).getByRole('region', { name: 'Used By' })).toContainText('1 of 2');
  await expect(steps.getByRole('button', { name: 'Select evaluate Evaluate 1 of 2', exact: true })).toBeVisible();
  await expect(evaluated.locator('[data-map-focus="downstream"]')).toHaveCount(1);

  await select(page, 'evaluate');
  const subset = drawer.getByRole('group', { name: 'Subject Subset', exact: true });
  await subset.getByRole('checkbox', { name: 'Fast · 0.003', exact: true }).check();
  const subsetEdit = edited('/inputs/subject/where');
  await subset.getByRole('button', { name: 'Apply Subset', exact: true }).click();
  const savedSubset = await subsetEdit;
  expect(savedSubset.ok(), await savedSubset.text()).toBe(true);
  await expect(inputs).toContainText('3 of 3');
  await expect(members.getByRole('listitem')).toHaveText(sourceRows.map(({ label }) => label));
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(2);
  const widened = (await readPlan()).window_map!.evaluations!;
  expect(widened.slice(0, 2)).toEqual(before);
  expect(widened).toHaveLength(3);
  await launchRows(page, projectPath, [widened[2].key]);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(3, { timeout: 16_000 });

  await page.reload();
  await select(page, 'evaluate');
  await expect(inputs).toContainText('3 of 3');
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(3);
  await runs(page).getByRole('row').filter({ hasText: 'Fast' }).getByRole('button').click();
  await expect(primary(page).getByRole('region', { name: 'Made From' })).toContainText('1 of 3');
  await expect(steps.getByRole('button', { name: 'Select train Train 1 of 3', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Show Map', exact: true }).click();
  await expect(train.locator('[data-map-focus="upstream"]')).toHaveCount(1);
  const final = await readPlan();
  const results = final.window_map!.evaluations!;
  expect(results.slice(0, 2)).toEqual(before);
  for (const row of results) {
    const edge = final.window_map!.edges.find(({ to_key }) => JSON.stringify(to_key) === JSON.stringify(row.key));
    const producer = final.rows.find(({ key }) => JSON.stringify(key) === JSON.stringify(edge?.from_key));
    expect(producer?.result).toBeDefined();
    expect(row.trials).toEqual(conditionIds);
    expect(row.result?.checkpoint_source).toEqual({
      job_id: producer!.result!.job_id, request_id: producer!.request_id, completed_updates: 2,
    });
    expect(row.result?.receipt.manifest_sha256).toMatch(/^[a-f0-9]{64}$/);
  }

  await page.getByRole('button', { name: 'Show Both', exact: true }).click();
  await select(page, 'train');
  await fast.click();
  await slow.click({ modifiers: ['Shift'] });
  const second = responseFor(page, projectPath + '/files');
  await primary(page).getByRole('button', { name: 'New Step', exact: true }).click();
  await page.getByRole('dialog', { name: 'New Step' }).getByRole('button', { name: 'Evaluate', exact: true }).click();
  const secondCreated = await second;
  expect(secondCreated.status(), await secondCreated.text()).toBe(201);
  const { snapshot: withSecond } = await secondCreated.json() as StudioProjectDocument;
  const picked = [sourceRows[0], sourceRows[2]];
  expect(withSecond.project.steps['evaluate-2'].inputs?.subject).toMatchObject({
    where: { rows: [authoredRows[0], authoredRows[2]] },
  });
  await expect(members.getByRole('listitem')).toHaveText(picked.map(({ label }) => label));
  await expect(trialSet.getByRole('combobox')).toHaveValue('every_condition');
  await select(page, 'evaluate');
  await expect(trialSet.getByRole('combobox')).toHaveValue('conditions');
  await select(page, 'evaluate-2');
  await expect(trialSet.getByRole('combobox')).toHaveValue('every_condition');
  await expect(trialSet.getByRole('button', { name: 'Apply Trials', exact: true })).toBeDisabled();
  await subset.getByRole('checkbox', { name: sourceRows[1].label, exact: true }).check();
  const widenRows = edited('/inputs/subject/where');
  await subset.getByRole('button', { name: 'Apply Subset', exact: true }).click();
  const widenedRows = await widenRows;
  expect(widenedRows.ok(), await widenedRows.text()).toBe(true);
  const widenedSnapshot = await widenedRows.json() as StudioProjectSnapshot;
  expect(widenedSnapshot.project.steps['evaluate-2'].inputs?.subject).toMatchObject({
    where: { rows: authoredRows },
  });
  await expect(members.getByRole('listitem')).toHaveText(sourceRows.map(({ label }) => label));
  console.info('Mounted Evaluate subset receipts', JSON.stringify({ project: snapshot.project_id, revision: final.revision, results }));
});

test('M2 pivots saved Evaluate Tasks and edits scoped Train Model fields', async ({ page, request }) => {
  test.setTimeout(75_000);
  await page.setViewportSize({ width: 1440, height: 900 });
  const files = JSON.parse(await runProjectFixture('web.e2e.studio.fixtures.canonical_evaluate', 'task-pivot-files'));
  const { snapshot, projectPath } = await seedManagedProject(request, files);
  const read = () => savedPlan(request, projectPath);
  await page.goto('/?project=' + snapshot.project_id);
  await select(page, 'evaluate');
  await primary(page).getByRole('button', { name: 'Pivot', exact: true }).click();
  const initial = await read();
  const train = initial.rows.filter(({ key }) => key.step === 'train');
  const rows = initial.window_map!.rows.filter(({ key }) => key.step === 'evaluate');
  expect(train).toHaveLength(6);
  expect(rows).toHaveLength(4);
  expect(initial.window_map?.problems).toHaveLength(4);
  expect(initial.window_map?.problems.every(({ step, path, code }) =>
    step === 'evaluate' && path === '/inputs/subject' && code === 'cannot_run')).toBe(true);
  expect(rows.map(({ coordinates }) => [coordinates['learning-rate'].label, coordinates.task.label])).toEqual([
    ['Slow', 'Short Task'], ['Slow', 'Long Task'], ['Fast', 'Short Task'], ['Fast', 'Long Task'],
  ]);
  const evaluation = initial.window_map!.evaluations!;
  expect(evaluation.map(({ trials }) => trials.length)).toEqual([2, 3, 2, 3]);

  const pivot = primary(page).getByRole('table', { name: 'Pivot', exact: true });
  await primary(page).getByRole('combobox', { name: 'Rows', exact: true }).selectOption('learning-rate');
  await primary(page).getByRole('combobox', { name: 'Columns', exact: true }).selectOption('task');
  const cell = (model: string, task: string, status: string) => pivot.getByRole('button', {
    name: `Select learning-rate ${model} · task ${task} · ${status}`, exact: true,
  });
  await expect(pivot.getByRole('columnheader')).toHaveText(['learning-rate', 'Short Task', 'Long Task']);
  await expect(pivot.getByRole('rowheader')).toHaveText(['Slow · 0.001', 'Fast · 0.003']);
  await expect(pivot.getByRole('button', { name: /Can't run$/ })).toHaveCount(4);

  await select(page, 'train');
  await launchRows(page, projectPath, train.map(({ key }) => key));
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(6, { timeout: 26_000 });
  const trainDone = (await read()).rows;
  await select(page, 'evaluate');
  await primary(page).getByRole('combobox', { name: 'Columns', exact: true }).selectOption('task');
  const sampled = cell('Fast · 0.003', 'Long Task', 'Not run');
  const target = rows[3].key;
  await sampled.click();
  await expect(sampled).toHaveAttribute('aria-pressed', 'true');
  await launchRows(page, projectPath, [target]);
  await expect(cell('Fast · 0.003', 'Long Task', 'Done')).toBeVisible({ timeout: 18_000 });
  await expect(pivot.getByRole('button', { name: /Not run$/ })).toHaveCount(3);
  const lineage = primary(page).getByRole('region', { name: 'Made From', exact: true });
  const complete = await read();
  expect(complete.window_map!.evaluations!.map(({ status }) => status)).toEqual([
    'not_run', 'not_run', 'not_run', 'succeeded',
  ]);
  const finished = complete.window_map!.evaluations!.find(({ key }) => key.row === target.row)!;
  const production = complete.window_map!.edges.find(({ to_key }) => to_key.row === target.row)!;
  const checkpoint = complete.rows.find(({ key }) => key.row === production.from_key.row)!;
  await expect(lineage).toContainText('1 of 6');
  await expect(lineage.getByRole('listitem').last()).toHaveText(checkpoint.label);
  expect(finished.result?.checkpoint_source).toEqual({
    job_id: checkpoint.result!.job_id, request_id: checkpoint.request_id, completed_updates: 2,
  });

  await select(page, 'train');
  await page.getByRole('button', { name: 'Split Details', exact: true }).click();
  const rail = page.getByRole('navigation', { name: 'Details', exact: true });
  await rail.getByRole('tab', { name: 'Model', exact: true }).click();
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await model.getByRole('article', { name: 'Plant', exact: true }).locator('header').click();
  const mass = page.getByRole('textbox', { name: 'Mass', exact: true });
  await expect(mass).toHaveAccessibleDescription('Document · reach-model');
  const rate = primary(page).getByRole('group', { name: 'learning-rate', exact: true });
  await rate.getByRole('button', { name: 'Fast · 0.003', exact: true }).click();
  await model.focus();
  await expect(mass).toBeEditable();
  await enter(mass, '2');
  await expect(runs(page).getByText('Out of date', { exact: true })).toHaveCount(2);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(4);
  await expect(mass).toHaveAccessibleDescription('learning-rate · Fast');
  const group = await read();
  const groupStale = group.rows.filter(({ status }) => status === 'stale');
  expect(groupStale.map(({ key }) => key.row)).toEqual(['learning-rate=Fast--seed=one', 'learning-rate=Fast--seed=two']);
  expect(groupStale.map(({ change }) => [change?.before, change?.after])).toEqual([[1, 2], [1, 2]]);

  const picked = trainDone.filter(({ key }) => key.row.startsWith('learning-rate=Slow'));
  const rowButton = (label: string) => runs(page).getByRole('button', { name: label, exact: true });
  await rowButton(picked[0].label).click();
  await rowButton(picked[1].label).click({ modifiers: ['Shift'] });
  await model.focus();
  await expect(mass).toBeEditable();
  await enter(mass, '3');
  await expect(runs(page).getByText('Out of date', { exact: true })).toHaveCount(4);
  await expect(mass).toHaveAccessibleDescription('Exception · field');
  const scoped = await read();
  const sweep = scoped.window_map!.steps.find(({ step_id }) => step_id === 'train')!;
  expect(sweep.exceptions).toHaveLength(1);
  expect(sweep.exceptions![0]).toMatchObject({ id: 'field', count: 2, rule: { where: { rows: picked.map(({ key }) => key.row) } } });
  expect(scoped.rows.filter(({ status }) => status === 'stale').map(({ key }) => key.row))
    .toEqual([...picked, ...groupStale].map(({ key }) => key.row));

  const current = await request.get(projectPath);
  const saved = await current.json() as StudioProjectSnapshot;
  const authorSeed = await request.patch(projectPath, { data: { revision: saved.revision, files: [{
    address: { kind: 'step', id: 'train' },
    patches: [{ op: 'add', path: '/sweep/axes/seed/values/three', value: {
      label: 'three', value: 3, patches: { 'method_payload.payload.seed': { value: 3 } },
    } }],
  }] } });
  expect(authorSeed.ok(), await authorSeed.text()).toBe(true);
  await page.reload();
  await select(page, 'train');
  const widened = await read();
  expect(widened.rows).toHaveLength(9);
  const newRows = widened.rows.filter(({ key }) => key.row.endsWith('seed=three'));
  expect(newRows.map(({ status }) => status)).toEqual(['not_run', 'not_run', 'not_run']);
  const sourceFor = async (row: StudioSavedTrainKey) => {
    const result = await request.post(projectPath + '/model/fields', { data: {
      revision: widened.revision, run: row, owner: { document: 'reach-model', composites: [], element: { kind: 'component', id: 'plant' } },
    } });
    expect(result.ok(), await result.text()).toBe(true);
    return (await result.json()).fields.find(({ label }) => label === 'Mass').sources;
  };
  expect(await sourceFor(newRows[2].key)).toMatchObject([{ kind: 'sweep_value', axis: 'learning-rate', value: 'Fast' }]);
  expect(await sourceFor(newRows[0].key)).toMatchObject([{ kind: 'document', name: 'reach-model' }]);
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  const exception = drawer.getByRole('region', { name: 'Sweep', exact: true })
    .getByRole('listitem').filter({ hasText: picked[0].key.row });
  await expect(exception).toContainText(`${picked[1].key.row} · 2`);
  await expect(exception).toContainText('Mass');
  await exception.getByRole('button', { name: 'Remove', exact: true }).click();
  await expect(runs(page).getByText('Out of date', { exact: true })).toHaveCount(2);
  await expect(runs(page).getByText('Done', { exact: true })).toHaveCount(4);
  await expect(exception).toHaveCount(0);
  const restored = (await read()).rows.filter(({ key }) => key.row.startsWith('learning-rate=Slow')).slice(0, 2);
  expect(restored).toEqual(picked);
});

test('renders analysis and a report, then reruns only results affected by an edit', async ({ page }) => {
  test.fixme(true, 'M3 · 041aba9, 3c83a61, 9b6cf79, 4d1558c, 6b22472');
  await page.goto('/');
  await expect(page.getByRole('option', { name: 'Analyze', exact: true })).toBeVisible();
});

test('uses a variant and a subset across the outline, map, and inspector', async ({ page }) => {
  test.fixme(true, 'M3 · 01e4c26, 3e0ceba');
  await page.goto('/');
  await expect(page.getByRole('button', { name: 'Pipeline Map', exact: true })).toBeVisible();
});
