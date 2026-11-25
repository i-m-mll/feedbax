import { expect, type Locator, type Page } from '@playwright/test';

/** The open Select or Combobox popup's list, apart from the listboxes drawn in the window. */
export function popupListbox(page: Page) {
  return page.getByRole('listbox').and(page.locator('[data-side] [role="listbox"], [role="listbox"][data-side]'));
}

/** Chooses `label` from a Select or Combobox of the control layer, by pointer. */
export async function pick(trigger: Locator, label: string | RegExp) {
  const page = trigger.page();
  await expect(trigger).toBeEnabled();
  await trigger.click();
  const listbox = popupListbox(page);
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

const detailsDrawer = (page: Page) => page.getByRole('complementary', { name: 'Details Drawer', exact: true });

/** Places a component from the Model drawer's Library by searching for it and pressing Return on its row. */
export async function placeFromLibrary(page: Page, name: string) {
  await detailsDrawer(page).getByRole('textbox', { name: 'Search Library', exact: true }).fill(name);
  await detailsDrawer(page).getByRole('option', { name, exact: true }).press('Enter');
}

/** A row of the Model drawer's Outline, by its element's name; a row's Problems count follows the name. */
export function outlineRow(page: Page, name: string) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return detailsDrawer(page).getByRole('tree', { name: 'Outline', exact: true })
    .getByRole('treeitem', { name: new RegExp(`^${escaped}( Problems \\d+)?$`) });
}

/** The bottom bar's trial picker, whose name gives the trial it shows. */
export function trialPicker(page: Page) {
  return page.locator('[data-studio-focus-region="bottom"]').getByRole('group', { name: 'Clock', exact: true })
    .getByRole('button', { name: /^Trial / });
}

/** The trial picker's popup, which chooses One, All or Mean and what each is of. */
export function trialPopup(page: Page) {
  return page.getByRole('dialog', { name: 'Trial', exact: true });
}

/** Shows the clock's trials as `kind` from the trial picker's popup, and closes it. */
export async function showTrials(page: Page, kind: 'One' | 'All' | 'Mean') {
  await trialPicker(page).click();
  await trialPopup(page).getByRole('group', { name: 'Trials', exact: true }).getByRole('button', { name: kind, exact: true }).click();
  await page.keyboard.press('Escape');
  await expect(trialPopup(page)).toBeHidden();
}

/** Picks the `index`-th trial the selected run recorded (from 0) from the clock's trial picker. */
export async function pickRecordedTrial(page: Page, index: number) {
  const picker = trialPicker(page);
  await picker.click();
  await page.getByRole('radio', { name: new RegExp(`^trial ${index + 1} of \\d+`) }).click();
  await expect(picker).toHaveText(new RegExp(`^trial ${index + 1} of \\d+$`));
}

/** The menu that asks where an edit is written. */
export function applyTo(page: Page) {
  return page.getByRole('menu', { name: 'Apply To', exact: true });
}

/**
 * The answer to the Model edit `gesture` makes. With `store`, the service is expected to ask where the edit applies, and
 * `store` is chosen in the Apply To menu; the answer is then that of the edit sent with the choice.
 */
export async function modelEdit(page: Page, projectPath: string, gesture: () => Promise<unknown>, store?: string) {
  const answer = () => page.waitForResponse((response) => response.request().method() === 'POST' &&
    new URL(response.url()).pathname === `${projectPath}/model/edit`);
  const [first] = await Promise.all([answer(), gesture()]);
  if (!store) return first;
  expect(first.ok(), await first.text()).toBe(true);
  expect((await first.json() as { result: { status: string } }).result.status).toBe('asked');
  const [chosen] = await Promise.all([answer(), applyTo(page).getByRole('menuitem', { name: store, exact: true }).click()]);
  return chosen;
}
