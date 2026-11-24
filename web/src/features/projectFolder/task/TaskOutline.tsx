import { studioName } from '@/components/activity/studioVocabulary';
import { StatusMark } from '@/components/ui/StatusMark';
import { Tree } from '@/components/ui/Tree';
import { Drawer, DrawerSection, type DrawerSections } from '@/features/projectWindow/DrawerForms';

/** The Outline's groups, in its order. */
const groupOrder = ['variables', 'epochs', 'signals', 'modelInputs'] as const;
export type OutlineGroup = (typeof groupOrder)[number];

/** An element of the task in the Outline: its name, and its first problem as its status mark. */
export interface OutlineEntry {
  readonly id: string;
  readonly name: string;
  readonly reason?: string;
  readonly muted?: boolean;
  /** Trailing text: what feeds a Model input, where its name is not the input's. */
  readonly detail?: string;
}

/**
 * The drawer's Outline: variables, epochs and signals, and the Model's inputs where the task feeds them, each element
 * with its status mark; a row selects its element in the view. A group with no elements is left out.
 */
export function TaskOutline({ sections, groups, selected, select }: {
  sections: DrawerSections;
  groups: Readonly<Partial<Record<OutlineGroup, readonly OutlineEntry[]>>>;
  selected?: { readonly group: OutlineGroup; readonly id: string };
  select(group: OutlineGroup, id: string): void;
}) {
  return <Drawer>
    {groupOrder.map((group) => {
      const entries = groups[group];
      if (!entries?.length) return null;
      const nodes = entries.map(({ id, name, reason, muted, detail }) => ({
        id, name, muted, trailing: reason || detail ? <>
          {detail && <span className="text-shell-muted">{detail}</span>}
          {reason && <StatusMark status="cannot_run" label={reason} />}
        </> : undefined,
      }));
      return <DrawerSection key={group} sections={sections} id={`task.${group}`} name={studioName(group)}>
        <Tree aria-label={studioName(group)} nodes={nodes} selected={selected?.group === group ? [selected.id] : []}
          onSelectedChange={(ids) => { const chosen = ids.at(-1); if (chosen) select(group, chosen); }} />
      </DrawerSection>;
    })}
  </Drawer>;
}
