import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { Activity, GripVertical, MoreHorizontal } from 'lucide-react';
import { useState, type DragEvent } from 'react';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { FieldProblem } from '@/components/ui/FieldProblem';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Table, TableCell, TableFieldCell, TableHeaderCell, TableRow, type TableColumn } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import { withAttention } from './attention';
import type { TaskSelection, TaskVariable } from './taskModel';

/** A refusal of a variable's edit, or nothing once it is made. */
type Made = Promise<string | undefined> | void;

export interface VariableEdits {
  /** Adds a variable named `name` at the end. */
  add?(name: string): Made;
  rename?(id: string, name: string): Made;
  remove?(id: string): Made;
  /** Moves the variable to `index` among the variables. */
  move?(id: string, index: number): Made;
}

/** A variable's type, with a list's count beside it. */
export function variableType({ type, count }: Pick<TaskVariable, 'type' | 'count'>) {
  return count === undefined ? type : `${type} × ${count}`;
}

/** The mark of a variable read every step, because it reads `time` or the model. */
export function LevelMark({ variable }: { variable: Pick<TaskVariable, 'level'> }) {
  if (variable.level !== 'step') return null;
  return <span role="img" aria-label={studioName('perStep')} {...appTip(studioName('perStep'))}
    className="inline-flex shrink-0 text-shell-muted [&_svg]:size-[1em]"><Activity aria-hidden="true" /></span>;
}

const columns: readonly TableColumn[] = [
  { least: '2rem', fixed: true }, { least: '9rem', fixed: true }, { least: '9rem', fixed: true }, { least: '10rem' },
];

const dragType = 'application/x-feedbax-task-variable';

/** The index a dragged variable lands at, dropped on the row at `index`: before it in its upper half, else after. */
function dropIndex(event: DragEvent<HTMLTableRowElement>, index: number) {
  const box = event.currentTarget.getBoundingClientRect();
  return event.clientY < box.top + box.height / 2 ? index : index + 1;
}

/**
 * Variables as a table of Name, Type and Value. A name is edited by clicking into it; a variable is added at the foot
 * of the Name column once it has a name, removed by its row's menu, and moved by dragging its grip.
 */
export function TaskVariables({ variables, selected, select, edits }: {
  variables: readonly TaskVariable[];
  selected?: TaskSelection;
  select(selection: TaskSelection): void;
  edits: VariableEdits;
}) {
  const [renaming, setRenaming] = useState<string>();
  const [adding, setAdding] = useState(false);
  const [refusals, setRefusals] = useState<Readonly<Record<string, string>>>({});
  const [dragging, setDragging] = useState<string>();
  const refuse = (id: string) => (reason: string | undefined) =>
    setRefusals((current) => ({ ...current, [id]: reason ?? '' }));
  const settle = (id: string, made: Made) => { void Promise.resolve(made).then((reason) => refuse(id)(reason || undefined)); };
  return <Table aria-label={studioName('variables')} columns={columns} held={2}
    addRow={edits.add && { label: studioName('addVariable'), onAdd: () => setAdding(true), disabled: adding }}>
    <thead><tr>
      <TableHeaderCell aria-hidden="true" />
      <TableHeaderCell scope="col">{studioName('name')}</TableHeaderCell>
      <TableHeaderCell scope="col">{studioName('type')}</TableHeaderCell>
      <TableHeaderCell scope="col">{studioName('value')}</TableHeaderCell>
    </tr></thead>
    <tbody>{variables.map((variable, index) => {
      const chosen = selected?.kind === 'variable' && selected.id === variable.id;
      const choose = () => { if (!chosen) select({ kind: 'variable', id: variable.id }); };
      const refusal = refusals[variable.id];
      return <TableRow key={variable.id} selected={chosen} className="group/owner"
        data-dragging={dragging === variable.id || undefined}
        onPointerDown={choose} onFocusCapture={choose}
        onDragOver={(event) => {
          if (!event.dataTransfer.types.includes(dragType)) return;
          event.preventDefault();
          event.dataTransfer.dropEffect = 'move';
        }}
        onDrop={(event) => {
          const id = event.dataTransfer.getData(dragType);
          if (!id || !edits.move) return;
          event.preventDefault();
          const from = variables.findIndex((item) => item.id === id);
          const to = dropIndex(event, index);
          const at = to > from ? to - 1 : to;
          if (at !== from) settle(id, edits.move(id, at));
        }}>
        <TableCell className="px-0 text-center">
          <span draggable={!!edits.move} aria-label={studioAccessibleName('move', variable.name)} role="img"
            className={clsx('inline-flex cursor-grab text-shell-muted [&_svg]:size-[1em]', withAttention)}
            onDragStart={(event) => {
              event.dataTransfer.setData(dragType, variable.id);
              event.dataTransfer.effectAllowed = 'move';
              setDragging(variable.id);
            }}
            onDragEnd={() => setDragging(undefined)}>
            <GripVertical aria-hidden="true" />
          </span>
        </TableCell>
        <TableFieldCell>
          <span className="flex h-full min-w-0 items-center">
            <InlineEdit form="cell" value={variable.name} editing={renaming === variable.id}
              className="min-w-0 flex-1 font-mono" aria-label={studioAccessibleName('rename', variable.name)}
              onEditingChange={(editing) => setRenaming(editing ? variable.id : undefined)}
              onCommit={(name) => { if (edits.rename) settle(variable.id, edits.rename(variable.id, name)); }} />
            {refusal && <FieldProblem reason={refusal} />}
            <LevelMark variable={variable} />
            {edits.remove && <Menu>
              <MenuTrigger render={<IconButton variant="ghost" size="mark" className={clsx('mr-1 shrink-0', withAttention)}
                label={studioAccessibleName('variable', variable.name)} icon={<MoreHorizontal aria-hidden="true" />} />} />
              <MenuContent aria-label={studioName('variable')}>
                <MenuItem onClick={() => setRenaming(variable.id)}>{studioName('rename')}</MenuItem>
                <MenuItem disabled={index === 0 || !edits.move}
                  onClick={() => settle(variable.id, edits.move?.(variable.id, index - 1))}>{studioName('moveEarlier')}</MenuItem>
                <MenuItem disabled={index === variables.length - 1 || !edits.move}
                  onClick={() => settle(variable.id, edits.move?.(variable.id, index + 1))}>{studioName('moveLater')}</MenuItem>
                <MenuItem onClick={() => settle(variable.id, edits.remove?.(variable.id))}>{studioName('remove')}</MenuItem>
              </MenuContent>
            </Menu>}
          </span>
        </TableFieldCell>
        <TableFieldCell {...appTip(variable.typeParts)}>
          <ValueField form="cell" model={variable.typeChoice} />
        </TableFieldCell>
        <TableFieldCell><ValueField form="cell" model={variable.value} /></TableFieldCell>
      </TableRow>;
    })}
    {adding && edits.add && <NewVariable add={edits.add} done={() => setAdding(false)} />}
    </tbody>
  </Table>;
}

/** A variable being added: its name, which writes the variable when it is given. */
function NewVariable({ add, done }: { add(name: string): Made; done(): void }) {
  const [refusal, setRefusal] = useState<string>();
  return <TableRow>
    <TableCell />
    <TableFieldCell>
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
    <TableCell /><TableCell />
  </TableRow>;
}
