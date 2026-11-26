import { expect, type Locator, type Page } from '@playwright/test';
import { palette, startProject } from './studioKeyboard';

/** Creates a project with one new Train and returns its focused Model panel once its canvas is drawn. */
export async function newModel(page: Page) {
  await startProject(page);
  await palette(page, 'newStep', 'New Step');
  await expect(page.getByRole('menu').getByRole('menuitem', { name: 'Train', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  const model = page.getByRole('tabpanel', { name: 'Model', exact: true });
  await expect(model).toBeFocused();
  await expect(model.locator('.react-flow__node[data-id="@inputs"]')).toBeVisible();
  return model;
}

/** Adds a `name` card from the Library's Runtime group and returns it once it can be edited. */
export async function addFromLibrary(page: Page, model: Locator, name: string) {
  const cards = model.locator('.react-flow__node.draggable');
  const ids = () => cards.evaluateAll((nodes) => nodes.map((node) => (node as HTMLElement).dataset.id!));
  const before = await ids();
  const drawer = page.getByRole('complementary', { name: 'Details Drawer', exact: true });
  await drawer.getByRole('textbox', { name: 'Search Library', exact: true }).fill(name);
  await drawer.getByRole('listbox', { name: 'Runtime', exact: true }).getByRole('option', { name, exact: true }).dblclick();
  await expect(cards).toHaveCount(before.length + 1);
  return model.locator(`.react-flow__node[data-id="${(await ids()).find((id) => !before.includes(id))}"]`);
}

/** Runs `act` and expects the Model edit it sends to be applied; returns the edit's result. */
export async function edited(page: Page, act: () => Promise<void>) {
  const saved = page.waitForResponse((response) => response.request().method() === 'POST' && response.url().endsWith('/model/edit'));
  await act();
  const { result } = await (await saved).json();
  expect(result.status, result.problem?.detail).toBe('applied');
  return result as { changed: { element: { id?: string } }[] };
}
