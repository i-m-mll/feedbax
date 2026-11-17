import { expect, type Locator } from '@playwright/test';

/** Chooses `label` from a Select or Combobox of the control layer, by pointer. */
export async function pick(trigger: Locator, label: string | RegExp) {
  const page = trigger.page();
  await expect(trigger).toBeEnabled();
  await trigger.click();
  const listbox = page.getByRole('listbox');
  await listbox.getByRole('option', { name: label, exact: true }).click();
  await expect(listbox).toBeHidden();
}
