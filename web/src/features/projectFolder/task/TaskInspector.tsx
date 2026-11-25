import type { ReactNode } from 'react';
import { studioName, studioUnit, studioValueType } from '@/components/activity/studioVocabulary';
import { FieldRow } from '@/components/ui/FieldRow';
import { Inspector } from '@/components/ui/Inspector';
import { ValueField } from '@/components/values/ValueField';
import { BatchTrace, TrialTrace } from './SignalTrace';
import { LevelMark, variableType } from './TaskVariables';
import type { TaskDocumentView, TaskSelection, TaskVariable } from './taskModel';
import { drawnVariables, epochBoundaries, signalAggregate, signalSteps, type TrialShown } from './trialSample';

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return <FieldRow label={label}><span className="flex h-control items-center tabular-nums text-shell-muted">{children}</span></FieldRow>;
}

const round = (value: number) => Number(value.toPrecision(3));

/** A signal's trace over the shown trial or trials, with one trial's epoch boundaries. */
function Trace({ view, signal, shown }: { view: TaskDocumentView; signal: string; shown?: TrialShown }) {
  const name = view.signals.find(({ id }) => id === signal)?.name ?? signal;
  if (!shown) return null;
  switch (shown.kind) {
    case 'one': return <section aria-label={studioName('plot')} className="grid gap-1">
      <TrialTrace label={name} steps={signalSteps(shown.record, signal)} boundaries={epochBoundaries(shown.record)} duration={view.duration} />
    </section>;
    case 'all': {
      const aggregate = signalAggregate(shown.aggregate, signal);
      return aggregate ? <section aria-label={studioName('plot')} className="grid gap-1">
        <BatchTrace label={name} aggregate={aggregate} boundaries={[]} duration={view.duration} />
      </section> : null;
    }
  }
}

/** A variable in the Inspector: its value with the value's parts, and its type. */
export function VariableInspector({ variable, children }: { variable: TaskVariable; children?: ReactNode }) {
  return <Inspector name={<span className="flex items-center gap-1 font-mono">{variable.name}<LevelMark variable={variable} /></span>}
    type={variableType(variable)}>
    <ValueField model={variable.value} />
    <ValueField model={variable.typeChoice} />
    {children}
  </Inspector>;
}

/** The Inspector for what is selected in the Task view: its fields with their parts, and, with a trial shown, what it drew. */
export function TaskInspector({ view, selected, shown }: { view: TaskDocumentView; selected?: TaskSelection; shown?: TrialShown }) {
  if (!selected) return null;
  const drawn = drawnVariables(shown);
  switch (selected.kind) {
    case 'variable': {
      const variable = view.variables.find(({ id }) => id === selected.id);
      if (!variable) return null;
      return <VariableInspector variable={variable}>
        {drawn && <Fact label={shown?.kind === 'all' ? studioName('allTrials') : studioName('trial')}>{drawn[variable.id]}</Fact>}
      </VariableInspector>;
    }
    case 'epoch': case 'end': {
      const epoch = view.epochs.find(({ id }) => id === selected.id);
      if (!epoch) return null;
      const { least, nominal, greatest } = epoch.bounds;
      const steps = view.stepSeconds;
      const span = (value: number) => steps ? `${round(value)} ${studioUnit('second')} · ${Math.round(value / steps)} ${studioUnit('steps')}`
        : `${round(value)} ${studioUnit('second')}`;
      return <Inspector name={<span className="font-mono">{epoch.name}</span>} type={studioName(selected.kind === 'end' ? 'end' : 'epoch')}>
        <ValueField model={epoch.end} />
        {epoch.bounds.ends !== 'length' && !epoch.end.reason && <Fact label={studioName('length')}>
          {least === greatest ? span(nominal) : `${span(least)} – ${span(greatest)}`}
        </Fact>}
      </Inspector>;
    }
    case 'signal': case 'cell': {
      const signal = view.signals.find(({ id }) => id === (selected.kind === 'cell' ? selected.signal : selected.id));
      if (!signal) return null;
      const cell = selected.kind === 'cell' ? signal.cells.find(({ id }) => id === selected.id) : undefined;
      const epochs = cell && view.epochs.filter((_, index) => index >= view.epochs.findIndex(({ id }) => id === cell.first) &&
        index <= view.epochs.findIndex(({ id }) => id === cell.last));
      return <Inspector name={<span className="font-mono">{signal.name}</span>}
        type={epochs ? `${signal.type} · ${epochs.map(({ name }) => name).join(', ')}` : studioValueType(signal.valueType, signal.type)}>
        {cell?.kind === 'value' && <ValueField model={cell.value} />}
        <ValueField model={signal.absent} />
        <Trace view={view} signal={signal.id} shown={shown} />
      </Inspector>;
    }
    case 'repeat': {
      const repeat = view.repeats.find(({ id }) => id === selected.id);
      if (!repeat) return null;
      const names = (id: string) => view.epochs.find((epoch) => epoch.id === id)?.name;
      return <Inspector name={studioName('repeat')} type={`${names(repeat.first)}–${names(repeat.last)}`}>
        <ValueField model={repeat.count} />
      </Inspector>;
    }
  }
}
