import { useSyncExternalStore } from 'react';
import { studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import { Dialog } from '@/components/ui/Dialog';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { Select } from '@/components/ui/Select';
import {
  resolveAppearance, setAppearance, subscribeAppearance, type AppearanceChoice, type AppearanceValue,
} from '@/appearance/appearance';

const rows = {
  theme: { control: 'segments', values: { system: 'system', light: 'light', dark: 'dark' } },
  face: { control: 'select', values: { system: 'system', 'work-sans': 'faceWorkSans', 'space-grotesk': 'faceSpaceGrotesk' } },
  size: { control: 'segments', values: { 12: 'pixels12', 13: 'pixels13', 14: 'pixels14' } },
  density: { control: 'segments', values: { compact: 'compact', regular: 'regular' } },
} as const satisfies {
  [Choice in AppearanceChoice]: {
    control: 'segments' | 'select';
    values: Record<AppearanceValue<Choice>, StudioNameId>;
  }
};

const subscribe = (notify: () => void) => subscribeAppearance(window, notify);

function useAppearance<Choice extends AppearanceChoice>(choice: Choice) {
  return useSyncExternalStore(subscribe, () => resolveAppearance(choice, window.localStorage));
}

function SettingRow<Choice extends AppearanceChoice>({ choice }: { choice: Choice }) {
  const value = useAppearance(choice);
  const row = rows[choice];
  const items = Object.entries(row.values).map(([value, name]) => ({
    value: value as AppearanceValue<Choice>, label: studioName(name as StudioNameId),
  }));
  const change = (next: AppearanceValue<Choice>) => setAppearance(window, choice, next);
  const label = studioName(choice);
  return (
    <>
      <span aria-hidden="true" className="text-shell-muted">{label}</span>
      {row.control === 'select'
        ? <Select aria-label={studioName(choice)} items={items} value={value} onValueChange={change} />
        : <SegmentedControl aria-label={studioName(choice)} items={items} value={value} onValueChange={change} />}
    </>
  );
}

export function StudioSettings({ close }: { close(): void }) {
  return (
    <Dialog ariaLabel={studioName('settings')} onClose={close} manageFocus
      panelClassName="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-4 gap-y-2 p-4">
      {(Object.keys(rows) as AppearanceChoice[]).map((choice) => <SettingRow key={choice} choice={choice} />)}
    </Dialog>
  );
}
