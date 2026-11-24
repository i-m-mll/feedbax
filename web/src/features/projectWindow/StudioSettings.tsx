import { useQuery } from '@tanstack/react-query';
import { useId, useLayoutEffect, useRef, useState, useSyncExternalStore, type ReactNode } from 'react';
import { SettingRow, SettingsPage, type SettingsSection } from '@/components/ui/SettingsPage';
import { fetchExecutionTargets } from '@/api/jobs';
import { studioName, studioUnit, type StudioNameId } from '@/components/activity/studioVocabulary';
import { StepCard } from '@/components/pipeline/StudioPipelineMap';
import { Checkbox } from '@/components/ui/Checkbox';
import { FieldRow } from '@/components/ui/FieldRow';
import { NumberField } from '@/components/ui/NumberField';
import { SegmentedControl } from '@/components/ui/SegmentedControl';
import { Select } from '@/components/ui/Select';
import { Table, TableCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import { rawField } from '@/components/values/adapters/raw';
import type { JsonValue } from '@/generated/studioContracts';
import type { PlanRunView, PlanStepView } from '@/features/pipeline/planView';
import { setRemovalThreshold, useRemovalThreshold } from '@/features/projectFolder/runRemoval';
import {
  resolveAppearance, setAppearance, subscribeAppearance, type AppearanceChoice, type AppearanceValue,
} from '@/appearance/appearance';
import { studioShortcuts } from './studioCommands';

const appearanceRows = {
  theme: { control: 'segments', values: { system: 'system', light: 'light', dark: 'dark' } },
  face: { control: 'select', values: { system: 'system', 'work-sans': 'faceWorkSans', 'space-grotesk': 'faceSpaceGrotesk' } },
  size: { control: 'segments', values: { 12: 'pixels12', 13: 'pixels13', 14: 'pixels14' } },
  density: { control: 'segments', values: { compact: 'compact', regular: 'regular' } },
  card: {
    control: 'select',
    values: { flow: 'cardFlowName', 'flow-ui': 'cardFlowUiName', turbo: 'cardTurboName', quiet: 'cardQuietName', ring: 'cardRingName' },
  },
} as const satisfies {
  [Choice in AppearanceChoice]: {
    control: 'segments' | 'select';
    values: Record<AppearanceValue<Choice>, StudioNameId>;
  }
};

const subscribe = (notify: () => void) => subscribeAppearance(window, notify);

function useAppearance<Choice extends AppearanceChoice>(choice: Choice) {
  return useSyncExternalStore(subscribe, () => resolveAppearance(choice, window));
}

function AppearanceRow<Choice extends AppearanceChoice>({ choice }: { choice: Choice }) {
  const value = useAppearance(choice);
  const row = appearanceRows[choice];
  const items = Object.entries(row.values).map(([value, name]) => ({
    value: value as AppearanceValue<Choice>, label: studioName(name as StudioNameId),
  }));
  const change = (next: AppearanceValue<Choice>) => setAppearance(window, choice, next);
  const label = studioName(choice);
  return (
    <SettingRow label={label} search={items.map(({ label }) => label).join(' ')}>
      {row.control === 'select'
        ? <Select aria-label={studioName(choice)} items={items} value={value} onValueChange={change} />
        : <SegmentedControl aria-label={studioName(choice)} items={items} value={value} onValueChange={change} />}
    </SettingRow>
  );
}

const seeds = { id: 'seed', label: 'seed', values: [{ id: '0', label: '0' }, { id: '1', label: '1' }] };
const rates = { id: 'rate', label: 'rate', values: [{ id: 'low', label: '0.001' }, { id: 'high', label: '0.01' }] };
const sampleStates = ['succeeded', 'succeeded', 'running', 'not_run'] as const;
const sampleStep: PlanStepView = {
  id: 'reach', label: 'reach', kind: 'train', sweeps: [rates, seeds], inputs: [],
  runs: rates.values.flatMap((rate, r) => seeds.values.map((seed, s): PlanRunView => {
    const state = sampleStates[r * 2 + s];
    return { id: `${rate.id}-${seed.id}`, label: `${rate.label}/${seed.label}`, values: { rate: rate.id, seed: seed.id }, status: state, state, results: [] };
  })),
  statusCounts: { succeeded: 2, running: 1, not_run: 1 },
};
const sampleRuns = [['run-1', 0, 0.412], ['run-2', 1, 0.388]] as const;
const none = () => {};

/** The real components, drawn with fixed values, so an appearance change is seen as it is made. */
function Specimen() {
  const id = useId();
  const group = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => { group.current!.inert = true; }, []);
  return (
    <div ref={group} role="group" aria-label={studioName('specimen')} data-settings-specimen className="grid content-start gap-3">
      <StepCard step={sampleStep} selection={null} presentation={{}} applySelection={none} chooseStep={none}
        compact={false} dimmed={false} />
      <div className="grid gap-1">
        <ValueField model={rawField({
          id: `${id}-rate`, label: studioName('learningRate'), type: { kind: 'number' }, value: 0.003, change: none,
        })} />
        <FieldRow label={studioName('seed')}>
          <SegmentedControl aria-label={studioName('seed')} value="0" onValueChange={none}
            items={seeds.values.map(({ id, label }) => ({ value: id, label }))} />
        </FieldRow>
      </div>
      <Table aria-label={studioName('runs')} columns={[{ least: '4rem' }, { least: '3rem' }, { least: '4rem' }]}
        box={{ className: 'rounded-control border border-shell-divider' }}>
        <thead><tr>
          <TableHeaderCell scope="col">{studioName('run')}</TableHeaderCell>
          <TableHeaderCell scope="col">{studioName('seed')}</TableHeaderCell>
          <TableHeaderCell scope="col">{studioName('learningRate')}</TableHeaderCell>
        </tr></thead>
        <tbody>{sampleRuns.map(([run, seed, value], index) => (
          <TableRow key={run} selected={index === 0}>
            <TableCell>{run}</TableCell>
            <TableCell className="tabular-nums">{seed}</TableCell>
            <TableCell className="tabular-nums">{value}</TableCell>
          </TableRow>
        ))}</tbody>
      </Table>
    </div>
  );
}

function ThresholdRow() {
  const seconds = useRemovalThreshold();
  const [draft, setDraft] = useState<number | null>();
  const label = studioName('removalThreshold');
  return (
    <SettingRow label={label}>
      <NumberField aria-label={studioName('removalThreshold')} className="w-20" min={0} step={1} value={draft === undefined ? seconds : draft}
        onValueChange={setDraft}
        onValueCommitted={(next) => { setDraft(undefined); if (next !== null) setRemovalThreshold(next); }} />
      <span className="text-shell-muted">{studioUnit('second')}</span>
    </SettingRow>
  );
}

/** Each machine a run can be started on, with the devices it reports. */
function Machines() {
  const targets = useQuery({ queryKey: ['execution-targets', []], queryFn: ({ signal }) => fetchExecutionTargets([], signal) });
  return targets.data?.targets.map(({ profile }) => {
    const devices = profile.kind === 'local'
      ? profile.runtime?.devices.join(' · ') ?? ''
      : Object.entries(profile.resources).map(([name, value]) => `${name} ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`).join(' · ');
    return (
      <SettingRow key={profile.id} label={profile.label} search={devices}>
        {devices && <span className="text-shell-muted">{devices}</span>}
      </SettingRow>
    );
  });
}

/** Studio's Settings: the app's settings, then the open project's (`project`) when there is one. */
export function StudioSettings({ project }: { project?: ReactNode }) {
  const sections: SettingsSection[] = [
    {
      id: 'appearance', title: studioName('appearance'), rows: (
        <div className="studio-settings-appearance">
          <div className="shell-settings__rows">
            {(Object.keys(appearanceRows) as AppearanceChoice[]).map((choice) => <AppearanceRow key={choice} choice={choice} />)}
          </div>
          <Specimen />
        </div>
      ),
    },
    { id: 'runs', title: studioName('runs'), rows: <div className="shell-settings__rows"><ThresholdRow /></div> },
    {
      id: 'shortcuts', title: studioName('shortcuts'), rows: (
        <div className="shell-settings__rows">
          {studioShortcuts().map(({ name, keys }) => (
            <SettingRow key={`${name} ${keys}`} label={name}><kbd className="font-sans text-shell-muted">{keys}</kbd></SettingRow>
          ))}
        </div>
      ),
    },
    { id: 'machines', title: studioName('machines'), rows: <div className="shell-settings__rows"><Machines /></div> },
    ...project ? [{ id: 'project', title: studioName('thisProject'), rows: <div className="shell-settings__rows">{project}</div> }] : [],
  ];
  return <SettingsPage label={studioName('settings')} searchLabel={studioName('searchSettings')} sections={sections} />;
}

/** The open project's settings, which are written to the project. */
export function ProjectSettingRows({ settings, steps, change }: {
  settings: Readonly<Record<string, unknown>>;
  steps: readonly { id: string; label: string }[];
  change: (name: string, setting: 'keep_superseded' | 'run_only_on_request', value: JsonValue) => void;
}) {
  const [draft, setDraft] = useState<number | null>();
  const stored = settings.keep_superseded;
  const days = stored === 'none' ? 0 : typeof stored === 'number' ? stored : defaultSupersededDays;
  const held = new Set(Array.isArray(settings.run_only_on_request) ? settings.run_only_on_request as string[] : []);
  const keepLabel = studioName('keepSuperseded');
  const holdLabel = studioName('manualSteps');
  return (
    <>
      <SettingRow label={keepLabel}>
        <NumberField aria-label={studioName('keepSuperseded')} className="w-20" min={0} step={1} value={draft === undefined ? days : draft}
          onValueChange={setDraft}
          onValueCommitted={(next) => {
            setDraft(undefined);
            if (next !== null && next !== days) change(keepLabel, 'keep_superseded', next);
          }} />
        <span className="text-shell-muted">{studioUnit('day')}</span>
      </SettingRow>
      {steps.length > 0 && <SettingRow label={holdLabel} search={steps.map(({ label }) => label).join(' ')}>
        <div role="group" aria-label={studioName('manualSteps')} className="grid gap-1">
          {steps.map(({ id, label }) => (
            <Checkbox key={id} label={label} checked={held.has(id)} onCheckedChange={(checked) => {
              const next = checked ? [...held, id] : [...held].filter((step) => step !== id);
              change(holdLabel, 'run_only_on_request', steps.map((step) => step.id).filter((step) => next.includes(step)));
            }} />
          ))}
        </div>
      </SettingRow>}
    </>
  );
}

/** `DEFAULT_SUPERSEDED_DAYS` of `feedbax/persistence/reachability.py`, which applies while the project names none. */
const defaultSupersededDays = 14;
