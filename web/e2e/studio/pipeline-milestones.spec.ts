import { expect, test } from '@playwright/test';

test('runs one row and resumes at its checkpoint after a restart', async ({ page }) => {
  test.fixme(true, 'M1 · f6f0bdd, de9e2ac, 5737f56, f61e141');
  await page.goto('/');
  // Use the reaching project with no sweep, then watch live loss, stop and resume the same run.
  // Restart the backend and host, reload the window, and verify one completion and restored history.
  await page.getByRole('button', { name: 'Run', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Stop', exact: true })).toBeEnabled();
  await page.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.getByRole('button', { name: 'Resume', exact: true })).toBeEnabled();
});

test('reopens a project folder and evaluates two conditions on the training result', async ({ page }) => {
  test.fixme(true, 'M2 · 736786f, dbbfb13, 339d687, 01e4c26, ed09a0a');
  await page.goto('/');
  // Quit and reopen the project folder. Add an Evaluate step over the run with two conditions.
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await expect(page.getByRole('option', { name: 'Evaluate', exact: true })).toBeVisible();
});

test('renders analysis and a report, then reruns only results affected by an edit', async ({ page }) => {
  test.fixme(true, 'M3 · 041aba9, 3c83a61, 9b6cf79, 4d1558c, 6b22472');
  await page.goto('/');
  // Add two operations and one figure, render a report, and rerun the project.
  // Edit a parameter and check only its operation, figure and report are stale; undo restores Done.
  await page.getByRole('button', { name: 'New Step', exact: true }).click();
  await expect(page.getByRole('option', { name: 'Analyze', exact: true })).toBeVisible();
});

test('uses a variant and a subset across the outline, map, and inspector', async ({ page }) => {
  test.fixme(true, 'M3 · 01e4c26, 3e0ceba');
  await page.goto('/');
  // Edit a variant without affecting base results. Sweep three rates and evaluate two selected runs.
  // Expand the subset to three and check one new run, reuse of the other two, and visible lineage.
  await expect(page.getByRole('button', { name: 'Pipeline Map', exact: true })).toBeVisible();
});
