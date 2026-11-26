import type { Locator, Page } from '@playwright/test';
import { savedRunId } from '../../../src/features/projectFolder/projectWindowPlan';

// Studio members found by key, not by the names the Service gives them, so a spec whose claim is not about naming does
// not depend on the naming rule. A spec that needs a member's visible text gives the member an authored label in its
// fixture instead.

type Scope = Page | Locator;
type RunKey = { readonly variant: string; readonly step: string; readonly row: string };

/** A run's row in a Runs table. */
export const runRow = (scope: Scope, key: RunKey) => scope.locator(`[data-run='${savedRunId(key)}']`);

/** A run's select button in a Runs table. */
export const runButton = (scope: Scope, key: RunKey) => runRow(scope, key).getByRole('rowheader').getByRole('button');

/** A step's row in the map drawer's Steps tree. */
export const stepRow = (scope: Scope, step: string) => scope.locator(`[role="treeitem"][data-node="${step}"]`);

/** A sweep's label on a map card. */
export const sweepButton = (card: Locator, axis: string) => card.locator(`[data-sweep="${axis}"]`);

/** A value in an open sweep menu. */
export const sweepValue = (page: Page, value: string) => page.getByRole('menuitem').and(page.locator(`[data-value="${value}"]`));

/** A document picker's trigger; `document` narrows it to the picker showing that document. */
export const documentPicker = (scope: Scope, document?: string) =>
  scope.locator(document === undefined ? 'button[data-document]' : `button[data-document="${document}"]`);

/** A document among an open picker menu's choices. */
export const documentChoice = (menu: Locator, document: string) => menu.locator(`[role="menuitemradio"][data-value="${document}"]`);

/** An operation or figure card on an Analysis canvas. */
export const canvasElement = (canvas: Locator, element: string) => canvas.locator(`.react-flow__node[data-id="${element}"]`);

/** A figure's button in a Figures view's list. */
export const figureButton = (figures: Locator, element: string) => figures.locator(`button[data-element="${element}"]`);

/** A command's control, such as a map card's New Step; `instance` narrows it to the command on that member. */
export const command = (scope: Scope, id: string, instance?: string) =>
  scope.locator(instance === undefined ? `[data-command-id="${id}"]` : `[data-command-id="${id}"][data-command-instance="${instance}"]`);

/** A Problems list's group of a step or document, as `step:<key>` or `document:<key>`. */
export const problemGroup = (scope: Scope, holder: `${'step' | 'document'}:${string}`) => scope.locator(`section[data-holder="${holder}"]`);

/** An Activity entry of a job. */
export const activityJob = (scope: Scope, job: string) => scope.locator(`li[data-job="${job}"]`);

/** The name a step's map card shows, for a spec that checks another place names the step as the map does. */
export const mapName = async (page: Page, step: string) =>
  (await page.locator(`[data-map-step="${step}"] .studio-map-card__head`).innerText()).trim();

/** The key of the member of a keyed collection whose label is `label`. */
export const keyLabelled = (members: Readonly<Record<string, { readonly label?: string }>> | undefined, label: string) =>
  Object.entries(members ?? {}).find(([, member]) => member.label === label)?.[0];
