import { expect, test, type APIRequestContext, type Locator, type Page } from '@playwright/test';
import { studioName } from '../../../src/components/activity/studioVocabulary';
import type { ProjectWindowView } from '../../../src/features/projectWindow/projectWindowState';
import type { StudioProjectDocument, StudioProjectSnapshot } from '../../../src/generated/studioContracts';
import { popupListbox } from './controls';

export async function tabTo(page: Page, target: Locator, limit = 100) {
  await expect(target).toBeVisible();
  const region = await target.evaluate((element) => element.closest('[role="dialog"]') ? null :
    element.closest<HTMLElement>('[data-studio-focus-region]')?.dataset.studioFocusRegion ?? null);
  if (region) {
    const focusedRegion = () => page.evaluate(() =>
      document.activeElement?.closest<HTMLElement>('[data-studio-focus-region]')?.dataset.studioFocusRegion);
    for (let index = 0; index < 8 && await focusedRegion() !== region; index += 1) {
      await page.keyboard.press('F6');
    }
    expect(await focusedRegion()).toBe(region);
  }
  const reverse = await target.evaluate((element) => {
    const active = document.activeElement;
    return !!active && active !== document.body &&
      !!(active.compareDocumentPosition(element) & Node.DOCUMENT_POSITION_PRECEDING);
  });
  for (let index = 0; index < limit; index += 1) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await page.keyboard.press(reverse ? 'Shift+Tab' : 'Tab');
  }
  await expect(target).toBeFocused();
}

export async function activateMenuItem(page: Page, item: Locator, key = 'Enter') {
  await expect(item).toBeEnabled();
  // A menu opened from the keyboard places its first focus a frame after it opens, which can take focus off an item
  // the loop already reached; reaching it again after that settles.
  await expect(async () => {
    for (let index = 0; index < 20 && !await item.evaluate((element) => element === document.activeElement); index += 1) {
      await page.keyboard.press('ArrowDown');
    }
    await expect(item).toBeFocused({ timeout: 250 });
  }).toPass();
  await page.keyboard.press(key);
}

export async function activate(page: Page, target: Locator, key = 'Enter') {
  await expect(target).toBeEnabled();
  await tabTo(page, target);
  await page.keyboard.press(key);
}

/** Chooses a kind in the open step menu; by keyboard, from the item the menu starts on. */
export async function chooseStepKind(page: Page, kind: string, keyboard = false) {
  const item = page.getByRole('menu').getByRole('menuitem', { name: kind, exact: true });
  if (keyboard) return activateMenuItem(page, item);
  await expect(item).toBeEnabled();
  await item.click();
}

export async function palette(page: Page, id: string, label: string) {
  await page.keyboard.press('Meta+K');
  const dialog = page.getByRole('dialog', { name: 'Command Palette' });
  await expect(dialog.getByRole('combobox')).toBeFocused();
  await page.keyboard.type(label);
  const item = dialog.getByRole('option').and(dialog.locator(`[data-command-id="${id}"]`));
  await expect(item).toHaveAccessibleName(new RegExp(`^${label}(?: |$)`));
  await expect(item).toHaveAttribute('data-command-id', id);
  await expect(item).toBeEnabled();
  const options = await dialog.getByRole('option').count();
  for (let index = 0; index < options && await item.getAttribute('aria-selected') !== 'true'; index += 1) {
    await page.keyboard.press('ArrowDown');
  }
  await expect(item).toHaveAttribute('aria-selected', 'true');
  await page.keyboard.press('Enter');
  await expect(dialog).toBeHidden();
}

export async function openView(page: Page, view: ProjectWindowView) {
  await palette(page, view, studioName(view));
  await expect(page.getByRole('tabpanel', { name: studioName(view), exact: true })).toBeVisible();
}

export async function typeValue(page: Page, target: Locator, value: string) {
  await expect(target).toBeEditable();
  await tabTo(page, target);
  // A saved edit can refresh the field projection while focus moves.
  await expect(target).toBeEditable();
  await expect(target).toBeFocused();
  await page.keyboard.press('ControlOrMeta+A');
  await page.keyboard.type(value);
  await expect(target).toHaveValue(value);
}

export async function number(page: Page, target: Locator, value: string) {
  await typeValue(page, target, value);
  await page.keyboard.press('Enter');
}

/** Submits a Select's option by keyboard. A scoped edit can ask where to store it after submission. */
export async function submitOption(page: Page, select: Locator, label: string | RegExp) {
  await expect(select).toBeEnabled();
  await tabTo(page, select);
  await page.keyboard.press('ArrowDown');
  const listbox = popupListbox(page);
  await expect(listbox).toBeVisible();
  const option = listbox.getByRole('option', { name: label, exact: true });
  await expect(option).toHaveCount(1);
  await page.keyboard.press('Home');
  const highlighted = listbox.locator('[role="option"][data-highlighted]');
  const options = await listbox.getByRole('option').count();
  for (let index = 0; index < options && !await option.evaluate((element) => element.hasAttribute('data-highlighted')); index += 1) {
    const before = await highlighted.count() ? await highlighted.getAttribute('id') : null;
    await page.keyboard.press('ArrowDown');
    if (before === null) await expect(highlighted).toHaveCount(1);
    else await expect(highlighted).not.toHaveAttribute('id', before);
  }
  await expect(option).toHaveAttribute('data-highlighted');
  await page.keyboard.press('Enter');
  await expect(listbox).toBeHidden();
}

/** Chooses the option and waits for the field to show it, taking an Apply To choice when supplied. */
export async function chooseOption(page: Page, select: Locator, label: string | RegExp, store?: string) {
  await submitOption(page, select, label);
  if (store) await activateMenuItem(page, page.getByRole('menu', { name: 'Apply To', exact: true }).getByRole('menuitem', { name: store, exact: true }));
  await expect(select).toHaveText(label);
}

export function post(page: Page, path: string) {
  return page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

export async function readSaved(request: APIRequestContext, projectPath: string) {
  const response = await request.get(projectPath);
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioProjectSnapshot;
}

/** Selects a row of the Model drawer's Outline by keyboard: Tab into the tree, then type the row's first word. */
export async function chooseOutlineRow(page: Page, row: Locator, name: string) {
  const tree = page.getByRole('complementary', { name: 'Details Drawer', exact: true }).getByRole('tree', { name: 'Outline', exact: true });
  await tabTo(page, tree.locator('[role="treeitem"][tabindex="0"]'));
  await page.keyboard.type(name.split(' ')[0]);
  await expect(row).toHaveAttribute('aria-selected', 'true');
}

/**
 * Makes a named project with the start page's New Project, which opens the new project's window, and returns the
 * project as its window opened it. The creating response's body is gone once the page navigates, so it is read again.
 */
export async function startProject(page: Page): Promise<StudioProjectDocument> {
  await page.goto('/');
  await page.getByRole('dialog', { name: 'Studio', exact: true }).getByRole('button', { name: 'New Project', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'New Project', exact: true });
  await dialog.getByRole('textbox', { name: 'Name', exact: true }).fill(testProjectName());
  const creation = page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === '/api/projects');
  await dialog.getByRole('button', { name: 'New Project', exact: true }).click();
  expect((await creation).status()).toBe(201);
  await page.waitForURL(/[?&]project=/);
  const id = new URL(page.url()).searchParams.get('project')!;
  const opened = await page.request.get(`/api/projects/${encodeURIComponent(id)}/open`);
  expect(opened.ok(), await opened.text()).toBe(true);
  return await opened.json() as StudioProjectDocument;
}

let projectSequence = 0;

export function testProjectName() {
  const title = test.info().title.toLowerCase().replace(/[^a-z0-9]+/g, '-').slice(0, 55).replace(/^-|-$/g, '');
  return `test-${title}-${++projectSequence}`;
}
