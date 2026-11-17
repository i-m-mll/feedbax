import { expect, type Locator, type Page } from '@playwright/test';

export async function pick(page: Page, trigger: Locator, label: string) {
  await trigger.click();
  await page.getByRole('listbox').getByRole('option', { name: label, exact: true }).click();
  await expect(trigger).toHaveText(label);
}
