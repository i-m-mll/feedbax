import { expect, test } from './fixtures/studio';
import { type Locator } from '@playwright/test';
import { fixture, mapStep, openResultsSlot, seedManagedProject } from './fixtures/canonicalTrain';
import type { StudioJobEventHistory, StudioJobSubmission, StudioTrainMetricHistory } from '../../src/generated/studioContracts';

/** The update under each vertex of a plotted curve, read from the x-axis labels the vertex sits over. */
function plottedUpdates(chart: Locator) {
  return chart.evaluate((svg) => {
    const ticks = Array.from(svg.querySelectorAll('text'), (text) => ({ x: Number(text.getAttribute('x')), label: text.textContent }));
    const curve = svg.querySelector('polyline');
    return Array.from({ length: curve?.points.numberOfItems ?? 0 }, (_, index) => {
      const { x } = curve!.points.getItem(index);
      return Number(ticks.find((tick) => Math.abs(tick.x - x) < 0.001)?.label);
    });
  });
}

test('a completed Train plots every update it retained', { tag: '@round' }, async ({ page, request }) => {
  const { snapshot, projectPath } = await seedManagedProject(request, await fixture('ordinary-files', '3'));
  const project = snapshot.project_id;
  const admission = await request.post(`${projectPath}/train-runs`, { data: {
    token: 'metric-history', revision: snapshot.revision,
    run_keys: [{ variant: 'default', step: 'train', row: 'default' }],
  } });
  expect(admission.status(), await admission.text()).toBe(202);
  const { items: [{ job_id: jobId }] } = await admission.json() as StudioJobSubmission;

  await page.goto(`/?project=${encodeURIComponent(project)}`);
  await mapStep(page, 'train').click();
  const table = page.getByRole('table', { name: 'Runs' });
  await expect(table.getByText('Done', { exact: true })).toBeVisible({ timeout: 20_000 });
  const retained = await request.get(`/api/jobs/${jobId}/training-metrics?project=${encodeURIComponent(project)}`);
  expect(retained.ok(), await retained.text()).toBe(true);
  const { updates } = await retained.json() as StudioTrainMetricHistory;
  expect(updates.map(({ update }) => update)).toEqual([1, 2, 3]);
  // Withhold the progress metrics, so whatever the chart plots is the Train's retained history.
  await page.route('**/api/events/history?**', async (route) => {
    const response = await route.fetch();
    const history = await response.json() as StudioJobEventHistory;
    const events = history.events.map((event) => {
      if (event.kind !== 'progress') return event;
      const { metrics: _metrics, metric_interval: _interval, ...data } = event.data;
      return { ...event, data };
    });
    await route.fulfill({ response, json: { ...history, events } });
  });
  await page.reload();
  await mapStep(page, 'train').click();

  await table.getByRole('button').first().click();
  await openResultsSlot(page);
  const results = page.getByRole('tabpanel', { name: 'Results', exact: true });
  const loss = results.getByRole('img', { name: 'Metric train_loss', exact: true });
  await expect.poll(() => plottedUpdates(loss)).toEqual([1, 2, 3]);

  await results.getByRole('button', { name: 'Current Result', exact: true }).click();
  const current = loss.locator(`[data-result-attempt="${jobId}"] line[data-update][data-value]`);
  await expect.poll(() => current.evaluateAll((points) => points.map((point) =>
    [Number(point.getAttribute('data-update')), Number(point.getAttribute('data-value'))])))
    .toEqual(updates.map(({ update, metrics }) => [update, metrics.train_loss]));
  await expect.poll(() => plottedUpdates(loss)).toEqual([1, 2, 3]);
});
