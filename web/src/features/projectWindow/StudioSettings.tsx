import { useState, useSyncExternalStore } from 'react';
import { studioName, studioUnit, type StudioNameId } from '@/components/activity/studioVocabulary';
import { NumberField } from '@/components/ui/NumberField';
import { setRemovalThreshold, useRemovalThreshold } from '@/features/projectFolder/runRemoval';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { Select } from '@/components/ui/Select';
import {
  resolveAppearance, setAppearance, subscribeAppearance, type AppearanceValue, type SettingChoice,
} from '@/appearance/appearance';

const rows = {
  theme: { control: 'segments', values: { system: 'system', light: 'light', dark: 'dark' } },
  face: { control: 'select', values: { system: 'system', 'work-sans': 'faceWorkSans', 'space-grotesk': 'faceSpaceGrotesk' } },
  size: { control: 'segments', values: { 12: 'pixels12', 13: 'pixels13', 14: 'pixels14' } },
  density: { control: 'segments', values: { compact: 'compact', regular: 'regular' } },
  card: {
    control: 'select',
    values: { flow: 'cardFlowName', 'flow-ui': 'cardFlowUiName', turbo: 'cardTurboName', quiet: 'cardQuietName', ring: 'cardRingName' },
  },
} as const satisfies {
  [Choice in SettingChoice]: {
    control: 'segments' | 'select';
    values: Record<AppearanceValue<Choice>, StudioNameId>;
  }
};

const subscribe = (notify: () => void) => subscribeAppearance(window, notify);

function useAppearance<Choice extends SettingChoice>(choice: Choice) {
  return useSyncExternalStore(subscribe, () => resolveAppearance(choice, window));
}

function SettingRow<Choice extends SettingChoice>({ choice }: { choice: Choice }) {
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

function ThresholdRow() {
  const seconds = useRemovalThreshold();
  const [draft, setDraft] = useState<number | null>();
  return (
    <>
      <span aria-hidden="true" className="text-shell-muted">{studioName('removalThreshold')}</span>
      <span className="flex items-center gap-1.5">
        <NumberField aria-label={studioName('removalThreshold')} className="w-20" min={0} step={1} value={draft === undefined ? seconds : draft}
          onValueChange={setDraft}
          onValueCommitted={(next) => { setDraft(undefined); if (next !== null) setRemovalThreshold(next); }} />
        <span className="text-shell-muted">{studioUnit('second')}</span>
      </span>
    </>
  );
}

export function StudioSettings() {
  return (
    <div className="grid w-fit grid-cols-[auto_minmax(0,1fr)] items-center gap-x-4 gap-y-2">
      {(Object.keys(rows) as SettingChoice[]).map((choice) => <SettingRow key={choice} choice={choice} />)}
      <ThresholdRow />
    </div>
  );
}
