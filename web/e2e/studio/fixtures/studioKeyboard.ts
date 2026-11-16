import { expect, type APIRequestContext, type Locator, type Page } from '@playwright/test';
import type { StudioProjectSnapshot } from '../../../src/generated/studioContracts';

export async function tabTo(page: Page, target: Locator, limit = 100) {
  await expect(target).toBeVisible();
  for (let index = 0; index < limit; index += 1) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await page.keyboard.press('Tab');
  }
  await expect(target).toBeFocused();
}

export async function activate(page: Page, target: Locator, key = 'Enter') {
  await expect(target).toBeEnabled();
  await tabTo(page, target);
  await page.keyboard.press(key);
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
  await page.keyboard.press('Enter');
  await expect(dialog).toBeHidden();
}

export async function number(page: Page, target: Locator, value: string) {
  await expect(target).toBeEditable();
  await tabTo(page, target);
  // A saved edit can refresh the field projection while focus moves.
  await expect(target).toBeEditable();
  await expect(target).toBeFocused();
  await page.keyboard.press('Meta+A');
  await page.keyboard.type(value);
  await expect(target).toHaveValue(value);
  await page.keyboard.press('Enter');
}

export function post(page: Page, path: string) {
  return page.waitForResponse((response) => response.request().method() === 'POST' && new URL(response.url()).pathname === path);
}

export async function readSaved(request: APIRequestContext, projectPath: string) {
  const response = await request.get(projectPath);
  expect(response.ok(), await response.text()).toBe(true);
  return await response.json() as StudioProjectSnapshot;
}
