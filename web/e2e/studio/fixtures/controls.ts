import { expect, type Locator, type Page } from '@playwright/test';

/** Chooses `label` from a Select or Combobox of the control layer, by pointer. */
export async function pick(trigger: Locator, label: string | RegExp) {
  const page = trigger.page();
  await expect(trigger).toBeEnabled();
  await trigger.click();
  const listbox = page.getByRole('listbox');
  await listbox.getByRole('option', { name: label, exact: true }).click();
  await expect(listbox).toBeHidden();
}

/** Shows a window panel from its status-bar toggle, whose label reports the panel's current state. */
export async function reveal(page: Page, panel: 'Drawer' | 'Inspector') {
  const footer = page.locator('[data-studio-focus-region="bottom"]');
  const show = footer.getByRole('button', { name: `Show ${panel}`, exact: true });
  const hide = footer.getByRole('button', { name: `Hide ${panel}`, exact: true });
  // Until the window fits its panes after a load, a panel stored open can read as hidden; a
  // click then stores it closed, so toggle until the panel is shown.
  await expect(async () => {
    if (await show.isVisible()) await show.click();
    await expect(hide).toBeVisible({ timeout: 1_000 });
  }).toPass({ timeout: 5_000 });
}
