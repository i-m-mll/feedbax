export const appearanceChoices = {
  theme: { initial: 'system', commands: { system: 'themeSystem', light: 'themeLight', dark: 'themeDark' } },
  face: {
    initial: 'work-sans',
    commands: { system: 'faceSystem', 'work-sans': 'faceWorkSans', 'space-grotesk': 'faceSpaceGrotesk' },
  },
  size: { initial: '13', commands: { 12: 'size12', 13: 'size13', 14: 'size14' } },
  density: { initial: 'regular', commands: { compact: 'densityCompact', regular: 'densityRegular' } },
} as const;

/**
 * The owner's design comparisons: root attributes switched from the command palette and overridable
 * per window by a query parameter of the same name. They are not settings.
 */
export const comparisonChoices = {
  unchecked: { initial: 'hollow', commands: { hollow: 'uncheckedHollow', badge: 'uncheckedBadge' } },
} as const;

type Comparisons = typeof comparisonChoices;
type ComparisonChoice = keyof Comparisons;
export type ComparisonCommandId = {
  [Choice in ComparisonChoice]: Comparisons[Choice]['commands'][keyof Comparisons[Choice]['commands']]
}[ComparisonChoice];
const comparisonNames = Object.keys(comparisonChoices) as ComparisonChoice[];
const comparisonKey = (choice: ComparisonChoice) => `feedbax.comparison.${choice}`;

function resolveComparison(choice: ComparisonChoice, view: Window): string {
  const commands = comparisonChoices[choice].commands;
  const valid = (value: string | null): value is string => value !== null && Object.prototype.hasOwnProperty.call(commands, value);
  const query = new URLSearchParams(view.location.search).get(choice);
  if (valid(query)) return query;
  const stored = view.localStorage.getItem(comparisonKey(choice));
  return valid(stored) ? stored : comparisonChoices[choice].initial;
}

export function comparisonSettings() {
  return comparisonNames.flatMap((choice) => Object.entries(comparisonChoices[choice].commands).map(([value, id]) => ({
    id: id as ComparisonCommandId,
    apply: (view: Window) => {
      view.localStorage.setItem(comparisonKey(choice), value);
      const url = new URL(view.location.href);
      if (url.searchParams.has(choice)) {
        url.searchParams.set(choice, value);
        view.history.replaceState(view.history.state, '', url);
      }
      applyAppearance(view);
    },
  })));
}

type Choices = typeof appearanceChoices;
export type AppearanceChoice = keyof Choices;
export type AppearanceValue<Choice extends AppearanceChoice> = keyof Choices[Choice]['commands'] & string;
export type AppearanceCommandId = {
  [Choice in AppearanceChoice]: Choices[Choice]['commands'][keyof Choices[Choice]['commands']]
}[AppearanceChoice];

const choiceNames = Object.keys(appearanceChoices) as AppearanceChoice[];
const storageKey = (choice: AppearanceChoice) => `feedbax.appearance.${choice}`;
const darkScheme = '(prefers-color-scheme: dark)';

function valid<Choice extends AppearanceChoice>(choice: Choice, value: string | null): value is AppearanceValue<Choice> {
  return value !== null && Object.prototype.hasOwnProperty.call(appearanceChoices[choice].commands, value);
}

const changeEvent = 'feedbax-appearance';

export function resolveAppearance<Choice extends AppearanceChoice>(
  choice: Choice, storage: Pick<Storage, 'getItem'>,
): AppearanceValue<Choice> {
  const stored = storage.getItem(storageKey(choice));
  return valid(choice, stored) ? stored : appearanceChoices[choice].initial as AppearanceValue<Choice>;
}

export function applyAppearance(view: Window) {
  const root = view.document.documentElement;
  for (const choice of choiceNames) {
    const value = resolveAppearance(choice, view.localStorage);
    if (choice !== 'theme') {
      root.setAttribute(`data-${choice}`, value);
      continue;
    }
    const theme = value === 'system' ? (view.matchMedia(darkScheme).matches ? 'dark' : 'light') : value;
    root.setAttribute('data-theme', theme);
    root.style.colorScheme = theme;
  }
  for (const choice of comparisonNames) root.setAttribute(`data-${choice}`, resolveComparison(choice, view));
}

export function installAppearance(view: Window) {
  applyAppearance(view);
  view.matchMedia(darkScheme).addEventListener('change', () => applyAppearance(view));
  view.addEventListener('storage', () => applyAppearance(view));
}

export function setAppearance<Choice extends AppearanceChoice>(view: Window, choice: Choice, value: AppearanceValue<Choice>) {
  view.localStorage.setItem(storageKey(choice), value);
  applyAppearance(view);
  view.dispatchEvent(new Event(changeEvent));
}

export function subscribeAppearance(view: Window, listener: () => void) {
  view.addEventListener(changeEvent, listener);
  view.addEventListener('storage', listener);
  return () => {
    view.removeEventListener(changeEvent, listener);
    view.removeEventListener('storage', listener);
  };
}

export function appearanceSettings() {
  return choiceNames.flatMap((choice) => Object.entries(appearanceChoices[choice].commands).map(([value, id]) => ({
    id: id as AppearanceCommandId,
    apply: (view: Window) => setAppearance(view, choice, value as AppearanceValue<typeof choice>),
  })));
}
