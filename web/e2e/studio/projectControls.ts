import { expect, type Page } from '@playwright/test';

export function openOverlay(page: Page) {
  const navigation = page.getByRole('navigation').filter({
    has: page.getByRole('button', { name: 'Projects', exact: true }),
  });
  return navigation.locator('xpath=..');
}

export async function openSavedProject(page: Page, name: string) {
  await page.getByRole('button', { name: 'Open project', exact: true }).click();
  const overlay = openOverlay(page);
  await overlay.getByRole('button', { name: new RegExp(`^${RegExp.escape(name)}(?: |$)`) }).click();
  await expect(overlay).toHaveCount(0);
  await expect(page.getByRole('banner').getByText(name, { exact: true }).first()).toBeVisible();
}
