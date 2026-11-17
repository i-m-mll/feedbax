export const appearanceChoices = {
  theme: { initial: 'system', commands: { system: 'themeSystem', light: 'themeLight', dark: 'themeDark' } },
  face: {
    initial: 'work-sans',
    commands: { system: 'faceSystem', 'work-sans': 'faceWorkSans', 'space-grotesk': 'faceSpaceGrotesk' },
  },
  size: { initial: '13', commands: { 12: 'size12', 13: 'size13', 14: 'size14' } },
  density: { initial: 'regular', commands: { compact: 'densityCompact', regular: 'densityRegular' } },
} as const;

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

export function resolveAppearance<Choice extends AppearanceChoice>(
  choice: Choice, search: string, storage: Pick<Storage, 'getItem'>,
): AppearanceValue<Choice> {
  const query = new URLSearchParams(search).get(choice);
  if (valid(choice, query)) return query;
  const stored = storage.getItem(storageKey(choice));
  return valid(choice, stored) ? stored : appearanceChoices[choice].initial as AppearanceValue<Choice>;
}

export function applyAppearance(view: Window) {
  const root = view.document.documentElement;
  for (const choice of choiceNames) {
    const value = resolveAppearance(choice, view.location.search, view.localStorage);
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
  if (url.searchParams.has(choice)) {
    url.searchParams.set(choice, value);
    view.history.replaceState(view.history.state, '', url);
  } else {
    view.localStorage.setItem(storageKey(choice), value);
  }
  applyAppearance(view);
}

export function appearanceSettings() {
  return choiceNames.flatMap((choice) => Object.entries(appearanceChoices[choice].commands).map(([value, id]) => ({
    id: id as AppearanceCommandId,
    apply: (view: Window) => setAppearance(view, choice, value as AppearanceValue<typeof choice>),
  })));
}
