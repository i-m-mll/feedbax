import { appTip } from '@window-shell/browser/tooltip';
import { Activity } from 'lucide-react';
import { useState } from 'react';
import { studioAccessibleName, studioName, studioValueType } from '@/components/activity/studioVocabulary';
import { ChangeMark } from '@/components/ui/ChangeMark';
import { FieldProblem } from '@/components/ui/FieldProblem';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { MenuCheckboxItem, MenuItem } from '@/components/ui/Menu';
import {
  Table, TableCell, TableFieldCell, TableHeaderCell, TableHeaderRow, TableRow, type TableColumn,
} from '@/components/ui/Table';
import { TreeChoice } from '@/components/ui/TreeChoice';
import { changeTip, ValueField } from '@/components/values/ValueField';
import type { TaskSelection, TaskVariable, UninitializedStart, VariableInitializes } from './taskModel';

/** A refusal of a variable's edit, or nothing once it is made. */
type Made = Promise<string | undefined> | void;

export interface VariableEdits {
  /** Adds a variable named `name` at the end; given `start`, it is of that value's type and initializes it. */
  add?(name: string, start?: UninitializedStart): Made;
  rename?(id: string, name: string): Made;
  remove?(id: string): Made;
  /** Moves the variable to `index` among the variables. */
  move?(id: string, index: number): Made;
  /** Makes the variable a list, or a single value again. */
  list?(id: string, list: boolean): Made;
}

/** A variable's type, with a list's count beside it. */
export function variableType({ type, count }: Pick<TaskVariable, 'type' | 'count'>) {
  return count === undefined ? type : `${type} × ${count.text ?? ''}`;
}

/** The mark of a variable read every step, because it reads `time` or the model. */
export function LevelMark({ variable }: { variable: Pick<TaskVariable, 'level'> }) {
  if (variable.level !== 'step') return null;
  return <span role="img" aria-label={studioName('perStep')} {...appTip(studioName('perStep'))}
    className="inline-flex shrink-0 text-shell-muted [&_svg]:size-[1em]"><Activity aria-hidden="true" /></span>;
}

const columns: readonly TableColumn[] = [{ least: '5rem', greatest: '16rem' }, { least: '6rem' }, { least: '8rem' }];
const drawnColumn: TableColumn = { least: '5rem' };
const initializesColumn: TableColumn = { least: '8rem' };

/** The Model values a variable initializes, as a choice over the Model's tree of those its type fits. */
function Initializes({ variable, initializes }: { variable: string; initializes?: VariableInitializes }) {
  const [refusal, setRefusal] = useState<string>();
  if (!initializes) return null;
  const { anchor, chosen, options, reason, changed, toggle } = initializes;
  const items = [...options, ...chosen.filter(({ id }) => !options.some((option) => option.id === id))];
  const shown = refusal ?? reason;
  return <div data-field={anchor} className="contents"><TreeChoice className="w-full" aria-label={studioAccessibleName('initializes', variable)}
    options={items.map(({ id, path }) => ({ value: id, path }))}
    checked={chosen.map(({ id }) => id)} text={chosen.map(({ name }) => name).join(', ')}
    disabled={!toggle} invalid={!!shown}
    trailing={(shown || changed) && <>
      {shown && <FieldProblem reason={shown} />}
      {changed && <ChangeMark tip={changeTip(changed.from)} reset={changed.reset && (() => { void changed.reset!(); })} />}
    </>}
    onToggle={(id) => {
      void Promise.resolve(toggle?.(id)).then((made) => setRefusal(made || undefined));
    }} /></div>;
}

/**
 * Variables as a table of Name, Type and Value. A name is edited by clicking into it; a variable is added at the foot
 * of the Name column once it has a name, removed by its row's menu, and moved by dragging its handle.
 */
export function TaskVariables({ variables, starts = [], selected, select, edits, drawn }: {
  variables: readonly TaskVariable[];
  /** The paired Model's values no variable initializes, each a row with its name empty. */
  starts?: readonly UninitializedStart[];
  selected?: TaskSelection;
  select(selection: TaskSelection): void;
  edits: VariableEdits;
  /** Each variable's value in the trial the clock shows, by id, drawn in a column before Initializes. */
  drawn?: Readonly<Record<string, string>>;
}) {
  const [renaming, setRenaming] = useState<string>();
  const [adding, setAdding] = useState(false);
  const [refusals, setRefusals] = useState<Readonly<Record<string, string>>>({});
  const refuse = (id: string) => (reason: string | undefined) =>
    setRefusals((current) => ({ ...current, [id]: reason ?? '' }));
  const settle = (id: string, made: Made) => { void Promise.resolve(made).then((reason) => refuse(id)(reason || undefined)); };
  const move = edits.move && ((id: string, index: number) => {
    const made = edits.move!(id, index);
    settle(id, made);
    return made;
  });
  return <Table aria-label={studioName('variables')} columns={[...columns, ...drawn ? [drawnColumn] : [], initializesColumn]} held={1}
    addRow={edits.add && { label: studioName('addVariable'), onAdd: () => setAdding(true), disabled: adding }}
    moveRow={move}>
    <thead><TableHeaderRow>
      <TableHeaderCell scope="col">{studioName('name')}</TableHeaderCell>
      <TableHeaderCell scope="col">{studioName('type')}</TableHeaderCell>
      <TableHeaderCell scope="col">{studioName('value')}</TableHeaderCell>
      {drawn && <TableHeaderCell scope="col">{studioName('drawn')}</TableHeaderCell>}
      <TableHeaderCell scope="col">{studioName('initializes')}</TableHeaderCell>
    </TableHeaderRow></thead>
    <tbody>{variables.map((variable, index) => {
      const chosen = selected?.kind === 'variable' && selected.id === variable.id;
      const choose = () => { if (!chosen) select({ kind: 'variable', id: variable.id }); };
      const refusal = refusals[variable.id];
      return <TableRow key={variable.id} selected={chosen} onPointerDown={choose} onFocusCapture={choose}
        sortable={move && { id: variable.id, label: studioAccessibleName('move', variable.name) }}
        menu={edits.remove && { label: studioAccessibleName('variable', variable.name), items: <>
          <MenuItem onClick={() => setRenaming(variable.id)}>{studioName('rename')}</MenuItem>
          <MenuItem disabled={index === 0 || !move} onClick={() => move?.(variable.id, index - 1)}>{studioName('moveEarlier')}</MenuItem>
          <MenuItem disabled={index === variables.length - 1 || !move}
            onClick={() => move?.(variable.id, index + 1)}>{studioName('moveLater')}</MenuItem>
          <MenuCheckboxItem checked={!!variable.count} disabled={!edits.list}
            onCheckedChange={(list) => settle(variable.id, edits.list?.(variable.id, list))}>{studioName('list')}</MenuCheckboxItem>
          <MenuItem onClick={() => settle(variable.id, edits.remove?.(variable.id))}>{studioName('remove')}</MenuItem>
        </> }}>
        <TableFieldCell header>
          <span className="flex h-full min-w-0 items-center">
            <InlineEdit form="cell" value={variable.name} editing={renaming === variable.id}
              className="min-w-0 flex-1 font-mono" aria-label={studioAccessibleName('rename', variable.name)}
              onEditingChange={(editing) => setRenaming(editing ? variable.id : undefined)}
              onCommit={(name) => { if (edits.rename) settle(variable.id, edits.rename(variable.id, name)); }} />
            {refusal && <FieldProblem reason={refusal} />}
            <LevelMark variable={variable} />
          </span>
        </TableFieldCell>
        <TableFieldCell {...appTip(studioValueType(variable.valueType))}>
          {variable.count ? <span className="flex min-w-0 items-center gap-1">
            <span className="min-w-0 flex-1"><ValueField form="cell" model={variable.typeChoice} /></span>
            <span aria-hidden="true" className="text-shell-muted">×</span>
            <span className="w-12 shrink-0"><ValueField form="cell" model={variable.count} /></span>
          </span> : <ValueField form="cell" model={variable.typeChoice} />}
        </TableFieldCell>
        <TableFieldCell><ValueField form="cell" model={variable.value} /></TableFieldCell>
        {drawn && <TableCell className="font-mono tabular-nums text-shell-muted">{drawn[variable.id]}</TableCell>}
        <TableFieldCell><Initializes variable={variable.name} initializes={variable.initializes} /></TableFieldCell>
      </TableRow>;
    })}
    {starts.map((start) => <StartRow key={start.id} start={start} add={edits.add} drawn={!!drawn} />)}
    {adding && edits.add && <NewVariable add={edits.add} done={() => setAdding(false)} drawn={!!drawn} />}
    </tbody>
  </Table>;
}

/** A variable being added: its name, which writes the variable when it is given. */
function NewVariable({ add, done, drawn }: { add(name: string): Made; done(): void; drawn: boolean }) {
  const [refusal, setRefusal] = useState<string>();
  return <TableRow>
    <TableFieldCell header>
      <span className="flex h-full min-w-0 items-center">
        <InlineEdit form="cell" value="" editing className="min-w-0 flex-1 font-mono"
          aria-label={studioName('addVariable')}
          onEditingChange={(editing) => { if (!editing && !refusal) done(); }}
          onCommit={(name) => {
            void Promise.resolve(add(name)).then((reason) => { setRefusal(reason || undefined); if (!reason) done(); });
          }} />
        {refusal && <FieldProblem reason={refusal} />}
      </span>
    </TableFieldCell>
    <TableCell /><TableCell />{drawn && <TableCell />}<TableCell />
  </TableRow>;
}

/**
 * A Model value no variable initializes: its type and the value, with the problem's mark when the Model needs it.
 * Naming it writes a variable of that type which initializes it.
 */
function StartRow({ start, add, drawn }: { start: UninitializedStart; add?: VariableEdits['add']; drawn: boolean }) {
  const [editing, setEditing] = useState(false);
  const [refusal, setRefusal] = useState<string>();
  const reason = refusal ?? start.reason;
  return <TableRow>
    <TableFieldCell header>
      <span data-field={start.anchor} className="flex h-full min-w-0 items-center">
        <InlineEdit form="cell" value="" editing={editing} className="min-w-0 flex-1 font-mono"
          aria-label={studioAccessibleName('name', start.name)} onEditingChange={setEditing}
          onCommit={(name) => {
            if (!add) return false;
            void Promise.resolve(add(name, start)).then((made) => setRefusal(made || undefined));
          }} />
      </span>
    </TableFieldCell>
    <TableCell className="text-shell-muted" {...appTip(studioValueType(start.valueType))}>{start.type}</TableCell>
    <TableCell />
    {drawn && <TableCell />}
    <TableCell>
      <span className="flex min-w-0 items-center">
        <span className="min-w-0 flex-1 truncate">{start.name}</span>
        {reason && <FieldProblem reason={reason} />}
      </span>
    </TableCell>
  </TableRow>;
}
