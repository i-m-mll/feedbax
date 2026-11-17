export const settingChoices = {
  theme: { initial: 'system', commands: { system: 'themeSystem', light: 'themeLight', dark: 'themeDark' } },
  face: {
    initial: 'work-sans',
    commands: { system: 'faceSystem', 'work-sans': 'faceWorkSans', 'space-grotesk': 'faceSpaceGrotesk' },
  },
  size: { initial: '13', commands: { 12: 'size12', 13: 'size13', 14: 'size14' } },
  density: { initial: 'regular', commands: { compact: 'densityCompact', regular: 'densityRegular' } },
} as const;

/** Design comparisons for the owner, switched from the command palette and overridable per window by query parameter. */
export const comparisonChoices = {
  unchecked: { initial: 'hollow', commands: { hollow: 'uncheckedHollow', badge: 'uncheckedBadge' } },
  'inspector-labels': { initial: 'beside', commands: { beside: 'inspectorLabelsBeside', above: 'inspectorLabelsAbove' } },
  'value-mark': { initial: 'hover', commands: { hover: 'valueMarkHover', quiet: 'valueMarkQuiet' } },
} as const;

export const appearanceChoices = { ...settingChoices, ...comparisonChoices } as const;

type Choices = typeof appearanceChoices;
export type AppearanceChoice = keyof Choices;
export type SettingChoice = keyof typeof settingChoices;
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

function compared(choice: AppearanceChoice) {
  return Object.prototype.hasOwnProperty.call(comparisonChoices, choice);
}

export function resolveAppearance<Choice extends AppearanceChoice>(
  choice: Choice, view: Pick<Window, 'localStorage' | 'location'>,
): AppearanceValue<Choice> {
  const query = compared(choice) ? new URLSearchParams(view.location.search).get(choice) : null;
  if (valid(choice, query)) return query;
  const stored = view.localStorage.getItem(storageKey(choice));
  return valid(choice, stored) ? stored : appearanceChoices[choice].initial as AppearanceValue<Choice>;
}

export function applyAppearance(view: Window) {
  const root = view.document.documentElement;
  for (const choice of choiceNames) {
    const value = resolveAppearance(choice, view);
    if (choice !== 'theme') {
      root.setAttribute(`data-${choice}`, value);
      continue;
    }
    const theme = value === 'system' ? (view.matchMedia(darkScheme).matches ? 'dark' : 'light') : value;
    root.setAttribute('data-theme', theme);
    root.style.colorScheme = theme;
  }
}

export function installAppearance(view: Window) {
  applyAppearance(view);
  view.matchMedia(darkScheme).addEventListener('change', () => applyAppearance(view));
  view.addEventListener('storage', () => applyAppearance(view));
}

export function setAppearance<Choice extends AppearanceChoice>(view: Window, choice: Choice, value: AppearanceValue<Choice>) {
  const url = new URL(view.location.href);
  if (compared(choice) && url.searchParams.has(choice)) {
    url.searchParams.set(choice, value);
    view.history.replaceState(view.history.state, '', url);
  } else {
    view.localStorage.setItem(storageKey(choice), value);
  }
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
