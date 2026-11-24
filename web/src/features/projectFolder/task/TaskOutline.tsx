import { studioName } from '@/components/activity/studioVocabulary';
import { StatusMark } from '@/components/ui/StatusMark';
import { Tree, type TreeNode } from '@/components/ui/Tree';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';
import type { FieldModel } from '@/components/values/fieldModel';
import type { TaskDocumentView, TaskSelection } from './taskModel';

/** The first problem among an element's fields, as its status mark. */
function mark(fields: readonly (FieldModel | undefined)[]) {
  const reason = fields.find((field) => field?.reason)?.reason;
  return reason ? <StatusMark status="cannot_run" label={reason} /> : undefined;
}

/** The drawer's Outline: variables, epochs and signals, each with its status mark; a row selects its element in the view. */
export function TaskOutline({ sections, view, selected, select }: {
  sections: DrawerSections; view: TaskDocumentView; selected?: TaskSelection; select(selection: TaskSelection): void;
}) {
  const groups = [
    { id: 'variables', kind: 'variable', nodes: view.variables.map((variable) => ({
      id: variable.id, name: variable.name, trailing: mark([variable.value]),
    })) },
    { id: 'epochs', kind: 'epoch', nodes: view.epochs.map((epoch) => ({ id: epoch.id, name: epoch.name, trailing: mark([epoch.end]) })) },
    { id: 'signals', kind: 'signal', nodes: view.signals.map((signal) => ({
      id: signal.id, name: signal.name, trailing: mark(signal.cells.map((cell) => cell.kind === 'value' ? cell.value : undefined)),
    })) },
  ] as const satisfies readonly { id: 'variables' | 'epochs' | 'signals'; kind: TaskSelection['kind']; nodes: readonly TreeNode[] }[];
  const owner = selected && (selected.kind === 'end' ? 'epoch' : selected.kind === 'cell' ? 'signal' : selected.kind);
  const ownerId = selected && (selected.kind === 'cell' ? selected.signal : selected.id);
  return <Drawer>
    {groups.map(({ id, kind, nodes }) => <DrawerSection key={id} sections={sections} id={`task.${id}`} name={studioName(id)}>
      <Tree aria-label={studioName(id)} nodes={nodes} selected={owner === kind && ownerId ? [ownerId] : []}
        onSelectedChange={(ids) => { const chosen = ids.at(-1); if (chosen) select({ kind, id: chosen }); }} />
    </DrawerSection>)}
  </Drawer>;
}
