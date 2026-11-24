import { appTip } from '@window-shell/browser/tooltip';
import clsx from 'clsx';
import { Activity, GripVertical, MoreHorizontal } from 'lucide-react';
import { useState } from 'react';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Table, TableCell, TableFieldCell, TableHeaderCell, TableRow } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import { withAttention } from './attention';
import type { TaskSelection, TaskVariable } from './taskModel';

export interface VariableEdits {
  rename?(id: string, name: string): void;
  remove?(id: string): void;
  move?(id: string, offset: -1 | 1): void;
  add?(): void;
}

/** A variable's type, with a list's count beside it. */
export function variableType({ type, count }: Pick<TaskVariable, 'type' | 'count'>) {
  return count === undefined ? type : `${type} × ${count}`;
}

/** The mark of a variable read every step, because it reads `time` or the model. */
export function LevelMark({ variable }: { variable: Pick<TaskVariable, 'level'> }) {
  if (variable.level !== 'step') return null;
  return <span role="img" aria-label={studioName('perStep')} {...appTip(studioName('perStep'))}
    className="inline-flex text-shell-muted [&_svg]:size-[1em]"><Activity aria-hidden="true" /></span>;
}

/**
 * Variables as a table of Name, Type and Value. With a trial shown, a last column holds the value it drew; over a batch,
 * the range of the values drawn.
 */
export function TaskVariables({ variables, drawn, drawnLabel, selected, select, edits }: {
  variables: readonly TaskVariable[];
  /** Each variable's drawn value, or range of values, by id, while a trial or a batch is shown. */
  drawn?: Readonly<Record<string, string>>;
  drawnLabel?: string;
  selected?: TaskSelection;
  select(selection: TaskSelection): void;
  edits: VariableEdits;
}) {
  const [renaming, setRenaming] = useState<string>();
  return <Table aria-label={studioName('variables')} held={2}
    columns={[{ least: '2rem', fixed: true }, { least: '9rem', fixed: true }, { least: '9rem', fixed: true }, { least: '10rem' },
      ...drawn ? [{ least: '10rem', fixed: true }] : []]}
    addRow={edits.add && { label: studioName('addVariable'), onAdd: edits.add }}>
    <thead><tr>
      <TableHeaderCell aria-hidden="true" />
      <TableHeaderCell scope="col">{studioName('name')}</TableHeaderCell>
      <TableHeaderCell scope="col">{studioName('type')}</TableHeaderCell>
      <TableHeaderCell scope="col">{studioName('value')}</TableHeaderCell>
      {drawn && <TableHeaderCell scope="col">{drawnLabel}</TableHeaderCell>}
    </tr></thead>
    <tbody>{variables.map((variable, index) => {
      const chosen = selected?.kind === 'variable' && selected.id === variable.id;
      return <TableRow key={variable.id} selected={chosen} className="group/owner"
        onPointerDown={() => { if (!chosen) select({ kind: 'variable', id: variable.id }); }}
        onFocusCapture={() => { if (!chosen) select({ kind: 'variable', id: variable.id }); }}>
        <TableCell className="!px-0">
          <span className={clsx('flex h-row items-center justify-center text-shell-muted', withAttention)}>
            <GripVertical aria-label={studioAccessibleName('moveLater', variable.name)} className="size-[1em] cursor-grab" />
          </span>
        </TableCell>
        <TableFieldCell>
          <span className="flex min-w-0 items-center gap-1 pr-1">
            <InlineEdit form="cell" value={variable.name} editing={renaming === variable.id} className="min-w-0 flex-1 font-mono"
              aria-label={studioAccessibleName('rename', variable.name)}
              onEditingChange={(editing) => setRenaming(editing ? variable.id : undefined)}
              onCommit={(name) => edits.rename?.(variable.id, name)} />
            <LevelMark variable={variable} />
            {edits.remove && <Menu>
              <MenuTrigger render={<IconButton variant="ghost" size="mark" className={clsx('ml-auto', withAttention)}
                label={studioName('variable')} icon={<MoreHorizontal aria-hidden="true" />} />} />
              <MenuContent aria-label={studioName('variable')}>
                <MenuItem onClick={() => setRenaming(variable.id)}>{studioName('rename')}</MenuItem>
                <MenuItem disabled={index === 0} onClick={() => edits.move?.(variable.id, -1)}>{studioName('moveEarlier')}</MenuItem>
                <MenuItem disabled={index === variables.length - 1} onClick={() => edits.move?.(variable.id, 1)}>{studioName('moveLater')}</MenuItem>
                <MenuItem onClick={() => edits.remove?.(variable.id)}>{studioName('remove')}</MenuItem>
              </MenuContent>
            </Menu>}
          </span>
        </TableFieldCell>
        <TableFieldCell>
          <span className="flex min-w-0 items-center gap-1">
            <span className="min-w-0 grow"><ValueField form="cell" model={variable.typeChoice} /></span>
            {variable.count !== undefined && <span className="shrink-0 pr-row-inline text-xs tabular-nums text-shell-muted">× {variable.count}</span>}
          </span>
        </TableFieldCell>
        <TableFieldCell><ValueField form="cell" model={variable.value} /></TableFieldCell>
        {drawn && <TableCell className="font-mono text-xs tabular-nums text-shell-muted">
          <span className="block truncate">{drawn[variable.id]}</span>
        </TableCell>}
      </TableRow>;
    })}</tbody>
  </Table>;
}
