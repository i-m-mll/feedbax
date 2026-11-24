import clsx from 'clsx';
import { GripHorizontal, MoreHorizontal, Plus } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Table, TableHeaderCell, TableRowHeader } from '@/components/ui/Table';
import { ValueField } from '@/components/values/ValueField';
import { withAttention } from './attention';
import { sameSelection, type TaskCell, type TaskEpoch, type TaskRepeat, type TaskSelection, type TaskSignal } from './taskModel';

export interface GridEdits {
  addEpoch?(before?: string): void;
  renameEpoch?(id: string, name: string): void;
  removeEpoch?(id: string): void;
  addSignal?(): void;
  /** Makes a cell absent, or gives an absent one a value. */
  toggleAbsent?(signal: string, cell: string): void;
  removeRepeat?(id: string): void;
  repeat?(first: string, last: string): void;
}

const cellBase = 'h-row border-b border-l border-shell-divider p-0 align-top';

/** An absent cell is its own state: hatched, with no value, and named Absent where 0 would be read as a value. */
function AbsentCell() {
  return <span role="img" aria-label={studioName('absent')}
    className="block h-row bg-[repeating-linear-gradient(135deg,var(--studio-control-border)_0_1px,transparent_1px_5px)]" />;
}

/** A spanning cell's edges, which a drag moves across adjacent columns. */
function SpanEdges() {
  return <>
    <span aria-hidden="true" className={clsx('absolute inset-y-1 left-0 w-1 cursor-ew-resize rounded-sm bg-shell-control', withAttention)} />
    <span aria-hidden="true" className={clsx('absolute inset-y-1 right-0 w-1 cursor-ew-resize rounded-sm bg-shell-control', withAttention)} />
  </>;
}

function Selectable({ selection, selected, select, className, colSpan, children }: {
  selection: TaskSelection; selected?: TaskSelection; select(selection: TaskSelection): void;
  className?: string; colSpan?: number; children: ReactNode;
}) {
  const chosen = sameSelection(selection, selected);
  return <td colSpan={colSpan} data-selected={chosen || undefined}
    className={clsx(cellBase, 'group/owner relative', chosen && 'bg-shell-row-selected outline outline-1 -outline-offset-1 outline-shell-foreground', className)}
    onPointerDown={() => { if (!chosen) select(selection); }} onFocusCapture={() => { if (!chosen) select(selection); }}>
    {children}
  </td>;
}

function EpochHeader({ epoch, selected, select, edits }: {
  epoch: TaskEpoch; selected?: TaskSelection; select(selection: TaskSelection): void; edits: GridEdits;
}) {
  const [renaming, setRenaming] = useState(false);
  const selection = { kind: 'epoch', id: epoch.id } as const;
  const chosen = sameSelection(selection, selected);
  return <TableHeaderCell scope="col" data-selected={chosen || undefined}
    className={clsx('group/owner relative border-l px-0', chosen && 'bg-shell-row-selected text-shell-foreground')}
    onPointerDown={() => { if (!chosen) select(selection); }}>
    <span className="flex h-row min-w-0 items-center gap-0.5 px-1" onDoubleClick={() => edits.renameEpoch && setRenaming(true)}>
      <GripHorizontal aria-label={studioAccessibleName('moveLater', epoch.name)}
        className={clsx('size-[1em] shrink-0 cursor-grab text-shell-muted', withAttention)} />
      <InlineEdit value={epoch.name} editing={renaming} onEditingChange={setRenaming} className="min-w-0 truncate font-mono"
        aria-label={studioAccessibleName('rename', epoch.name)} onCommit={(name) => edits.renameEpoch?.(epoch.id, name)} />
      {edits.removeEpoch && <Menu>
        <MenuTrigger render={<IconButton variant="ghost" size="mark" className={clsx('ml-auto', withAttention)}
          label={studioName('epoch')} icon={<MoreHorizontal aria-hidden="true" />} />} />
        <MenuContent aria-label={studioName('epoch')}>
          <MenuItem onClick={() => setRenaming(true)}>{studioName('rename')}</MenuItem>
          <MenuItem onClick={() => edits.addEpoch?.(epoch.id)}>{studioName('insertEpoch')}</MenuItem>
          <MenuItem onClick={() => edits.removeEpoch?.(epoch.id)}>{studioName('remove')}</MenuItem>
        </MenuContent>
      </Menu>}
    </span>
    {edits.addEpoch && <IconButton variant="ghost" size="mark" label={studioName('insertEpoch')} icon={<Plus aria-hidden="true" />}
      onClick={() => edits.addEpoch!(epoch.id)}
      className={clsx('absolute -left-2.5 top-1/2 z-10 -translate-y-1/2 rounded-full bg-shell-surface', withAttention)} />}
  </TableHeaderCell>;
}

/** The repeats over the columns, as a lane above them: a bracket over each, its count a field at its end. */
function RepeatLane({ epochs, repeats, selected, select }: {
  epochs: readonly TaskEpoch[]; repeats: readonly TaskRepeat[]; selected?: TaskSelection; select(selection: TaskSelection): void;
}) {
  const cells: ReactNode[] = [];
  for (let index = 0; index < epochs.length;) {
    const repeat = repeats.find(({ first }) => first === epochs[index].id);
    if (!repeat) {
      cells.push(<th key={epochs[index].id} aria-hidden="true" />);
      index += 1;
      continue;
    }
    const last = epochs.findIndex(({ id }) => id === repeat.last);
    const span = last - index + 1;
    const selection = { kind: 'repeat', id: repeat.id } as const;
    const chosen = sameSelection(selection, selected);
    cells.push(<th key={repeat.id} colSpan={span} scope="colgroup" aria-label={studioName('repeat')}
      className="px-1 pb-0.5 pt-1 font-normal" onPointerDown={() => { if (!chosen) select(selection); }}>
      <span className={clsx('flex items-end gap-1', chosen && 'text-shell-foreground')}>
        <span aria-hidden="true" className={clsx('h-2 grow border-x border-t', chosen ? 'border-shell-foreground' : 'border-shell-muted')} />
        <span className="flex w-16 shrink-0 items-center gap-0.5 text-xs text-shell-muted">×<ValueField form="cell" model={repeat.count} /></span>
      </span>
    </th>);
    index = last + 1;
  }
  return <tr><th aria-hidden="true" />{cells}<th aria-hidden="true" /></tr>;
}

function SignalCell({ signal, cell, epochs, selected, select }: {
  signal: TaskSignal; cell: TaskCell; epochs: readonly TaskEpoch[]; selected?: TaskSelection; select(selection: TaskSelection): void;
}) {
  const first = epochs.findIndex(({ id }) => id === cell.first);
  const last = epochs.findIndex(({ id }) => id === cell.last);
  const span = last - first + 1;
  return <Selectable selection={{ kind: 'cell', signal: signal.id, id: cell.id }} selected={selected} select={select} colSpan={span}>
    {cell.kind === 'absent' ? <AbsentCell /> : <div className="px-1.5 py-0.5"><ValueField form="cell" model={cell.value} /></div>}
    {span > 1 && <SpanEdges />}
  </Selectable>;
}

/**
 * Epochs as columns of equal width and signals as rows, under a lane of repeats. The End row holds each epoch's End;
 * a cell is a field, spans adjacent columns, or is absent.
 */
export function TaskGrid({ epochs, repeats, signals, selected, select, edits }: {
  epochs: readonly TaskEpoch[]; repeats: readonly TaskRepeat[]; signals: readonly TaskSignal[];
  selected?: TaskSelection; select(selection: TaskSelection): void; edits: GridEdits;
}) {
  return <Table aria-label={studioName('epochs')} className="table-fixed">
    <colgroup><col className="w-40" />{epochs.map(({ id }) => <col key={id} />)}<col className="w-8" /></colgroup>
    <thead>
      {repeats.length > 0 && <RepeatLane epochs={epochs} repeats={repeats} selected={selected} select={select} />}
      <tr>
        <TableHeaderCell aria-hidden="true" />
        {epochs.map((epoch) => <EpochHeader key={epoch.id} epoch={epoch} selected={selected} select={select} edits={edits} />)}
        <TableHeaderCell className="border-l px-0 text-center">
          {edits.addEpoch && <IconButton variant="ghost" size="mark" label={studioName('addEpoch')} icon={<Plus aria-hidden="true" />}
            onClick={() => edits.addEpoch!()} />}
        </TableHeaderCell>
      </tr>
    </thead>
    <tbody>
      <tr>
        <TableRowHeader className="text-shell-muted">{studioName('end')}</TableRowHeader>
        {epochs.map((epoch) => <Selectable key={epoch.id} selection={{ kind: 'end', id: epoch.id }} selected={selected} select={select}>
          <div className="px-1.5 py-0.5"><ValueField form="cell" model={epoch.end} /></div>
        </Selectable>)}
        <td className={cellBase} />
      </tr>
      {signals.map((signal) => {
        const selection = { kind: 'signal', id: signal.id } as const;
        const chosen = sameSelection(selection, selected);
        return <tr key={signal.id} className="group/owner" data-selected={chosen || undefined}>
          <TableRowHeader className={clsx('cursor-default', chosen && 'bg-shell-row-selected font-medium')}
            onPointerDown={() => { if (!chosen) select(selection); }}>
            <span className="flex min-w-0 items-baseline gap-1.5">
              <span className="truncate font-mono">{signal.name}</span>
              <span className="truncate text-xs text-shell-muted">{signal.type}</span>
            </span>
          </TableRowHeader>
          {signal.cells.map((cell) => <SignalCell key={cell.id} signal={signal} cell={cell} epochs={epochs} selected={selected} select={select} />)}
          <td className={cellBase} />
        </tr>;
      })}
      {edits.addSignal && <tr>
        <td className="px-1"><IconButton variant="ghost" size="mark" label={studioName('addSignal')} icon={<Plus aria-hidden="true" />}
          onClick={edits.addSignal} /></td>
      </tr>}
    </tbody>
  </Table>;
}
