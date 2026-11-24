import { followOtherWindows, readSettings, subscribeSettings, writeSetting } from './appSettings';

export const appearanceChoices = {
  theme: { initial: 'system', commands: { system: 'themeSystem', light: 'themeLight', dark: 'themeDark' } },
  face: {
    initial: 'work-sans',
    commands: { system: 'faceSystem', 'work-sans': 'faceWorkSans', 'space-grotesk': 'faceSpaceGrotesk' },
  },
  size: { initial: '13', commands: { 12: 'size12', 13: 'size13', 14: 'size14' } },
  density: { initial: 'regular', commands: { compact: 'densityCompact', regular: 'densityRegular' } },
  card: {
    initial: 'ring',
    commands: { flow: 'cardFlow', 'flow-ui': 'cardFlowUi', turbo: 'cardTurbo', quiet: 'cardQuiet', ring: 'cardRing' },
  },
} as const;

type Choices = typeof appearanceChoices;
export type AppearanceChoice = keyof Choices;
export type AppearanceValue<Choice extends AppearanceChoice> = keyof Choices[Choice]['commands'] & string;
export type AppearanceCommandId = {
  [Choice in AppearanceChoice]: Choices[Choice]['commands'][keyof Choices[Choice]['commands']]
}[AppearanceChoice];

const choiceNames = Object.keys(appearanceChoices) as AppearanceChoice[];
const darkScheme = '(prefers-color-scheme: dark)';

function valid<Choice extends AppearanceChoice>(choice: Choice, value: unknown): value is AppearanceValue<Choice> {
  return typeof value === 'string' && Object.prototype.hasOwnProperty.call(appearanceChoices[choice].commands, value);
}

export function resolveAppearance<Choice extends AppearanceChoice>(choice: Choice, view: Window): AppearanceValue<Choice> {
  const stored = readSettings(view)[choice];
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
  subscribeSettings(view, () => applyAppearance(view));
  followOtherWindows(view);
}

export function setAppearance<Choice extends AppearanceChoice>(view: Window, choice: Choice, value: AppearanceValue<Choice>) {
  void writeSetting(view, choice, value);
}

export function subscribeAppearance(view: Window, listener: () => void) {
  return subscribeSettings(view, listener);
}

export function appearanceSettings() {
  return choiceNames.flatMap((choice) => Object.entries(appearanceChoices[choice].commands).map(([value, id]) => ({
    id: id as AppearanceCommandId,
    apply: (view: Window) => setAppearance(view, choice, value as AppearanceValue<typeof choice>),
  })));
}
