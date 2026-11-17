import { expect, type APIRequestContext, type Locator, type Page } from '@playwright/test';
import { studioName } from '../../../src/components/activity/studioVocabulary';
import type { ProjectWindowView } from '../../../src/features/projectWindow/projectWindowState';
import type { StudioProjectSnapshot } from '../../../src/generated/studioContracts';

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

export async function activate(page: Page, target: Locator, key = 'Enter') {
  await expect(target).toBeEnabled();
  await tabTo(page, target);
  await page.keyboard.press(key);
}

/** Chooses a kind in the open step menu; by keyboard, from the item the menu starts on. */
export async function chooseStepKind(page: Page, kind: string, keyboard = false) {
  const item = page.getByRole('menu').getByRole('menuitem', { name: kind, exact: true });
  await expect(item).not.toHaveAttribute('aria-disabled', 'true');
  if (!keyboard) return item.click();
  for (let step = 0; step < 4 && !await item.evaluate((element) => element === document.activeElement); step += 1) {
    await page.keyboard.press('ArrowDown');
  }
  await expect(item).toBeFocused();
  await page.keyboard.press('Enter');
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

export async function chooseOption(page: Page, select: Locator, label: string | RegExp) {
  const option = select.getByRole('option', { name: label, exact: true });
  await expect(option).toHaveCount(1);
  const { keys, value } = await option.evaluate((element) => {
    const option = element as HTMLOptionElement;
    const choices = [...option.closest('select')!.options].filter((item) => !item.disabled);
    const name = option.label.trim().toLowerCase();
    let length = 1;
    while (length < name.length &&
      choices.filter((item) => item.label.trim().toLowerCase().startsWith(name.slice(0, length))).length > 1) {
      length += 1;
    }
    return { keys: name.slice(0, length), value: option.value };
  });
  await tabTo(page, select);
  await page.keyboard.type(keys);
  await expect(select).toHaveValue(value);
}

export function post(page: Page, path: string) {
  return page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

export async function readSaved(request: APIRequestContext, projectPath: string) {
  const response = await request.get(projectPath);
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioProjectSnapshot;
}
