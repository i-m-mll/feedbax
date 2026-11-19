import { Search } from 'lucide-react';
import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { ListRow } from '@/components/ui/ListRow';
import { TextField } from '@/components/ui/TextField';
import { StudioRegion, type StudioRegionState } from '@/components/activity/StudioRegion';
import { studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import { DrawerSection, type DrawerSections } from './DrawerForms';
import type { ProjectWindowSection } from './projectWindowState';

export interface LibraryItem {
  id: string;
  name: string;
  group: string;
  glyph?: ReactNode;
  /** Further text a search matches, beside the name and group. */
  terms?: string;
}

export interface LibraryListProps {
  /** The catalog's read; its items are drawn when it has a result. */
  state: StudioRegionState;
  items: readonly LibraryItem[];
  sections: DrawerSections;
  /** The key of the drawer section the list is; its groups are kept under it. */
  section: ProjectWindowSection;
  name: ReactNode;
  /** Names the search field. */
  searchName: StudioNameId;
  selected?: string;
  /** A click or the arrow keys: the item is shown, not placed. */
  onInspect: (id: string) => void;
  /** Return or a double-click places the item. */
  onPlace?: (id: string) => void;
  /** The drag's data type; an item is dragged with its id. */
  dragType?: string;
}

function matches(item: LibraryItem, query: string) {
  return `${item.name} ${item.group} ${item.terms ?? ''}`.toLowerCase().includes(query);
}

/** A drawer section holding a catalog: a search field over groups of rows, folded until opened or searched. */
export function LibraryList({
  state, items, sections, section, name, searchName, selected, onInspect, onPlace, dragType,
}: LibraryListProps) {
  const [query, setQuery] = useState('');
  const names = useId();
  const [focused, setFocused] = useState<string>();
  const rows = useRef(new Map<string, HTMLDivElement>());
  const search = query.trim().toLowerCase();
  const groups = new Map<string, LibraryItem[]>();
  for (const item of items) {
    if (search && !matches(item, search)) continue;
    const group = groups.get(item.group);
    if (group) group.push(item);
    else groups.set(item.group, [item]);
  }
  const groupKey = (group: string) => `${section}.${group}` as ProjectWindowSection;
  const shown = [...groups].flatMap(([group, list]) => search || sections.isOpen(groupKey(group), false) ? list : []);
  const current = shown.find(({ id }) => id === focused)?.id ?? shown.find(({ id }) => id === selected)?.id ?? shown[0]?.id;

  const focus = (id: string) => {
    setFocused(id);
    rows.current.get(id)?.focus();
    rows.current.get(id)?.scrollIntoView({ block: 'nearest' });
    onInspect(id);
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>, id: string) => {
    const at = shown.findIndex((item) => item.id === id);
    const next = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: shown.length - 1 }[event.key];
    if (next !== undefined) {
      event.preventDefault();
      if (shown[next]) focus(shown[next].id);
    } else if (event.key === 'Enter' && onPlace) {
      event.preventDefault();
      onPlace(id);
    }
  };

  const searchRegion = <div className="border-b border-[color:var(--neutral-alpha-06)] p-2">
    <TextField aria-label={studioName(searchName)} leading={<Search />} value={query} onValueChange={setQuery} />
  </div>;
  return <DrawerSection sections={sections} id={section} name={name} lead={searchRegion}>
    <StudioRegion state={state}>
      {[...groups].map(([group, list], index) => <DrawerSection key={group} sections={sections} id={groupKey(group)}
        level="group" name={group} nameId={`${names}-${index}`} initially={false} forceOpen={!!search}>
        <div role="listbox" aria-labelledby={`${names}-${index}`} className="grid gap-px px-1 pb-1">
          {list.map((item) => <ListRow key={item.id} role="option" name={item.name} glyph={item.glyph}
            ref={(element) => {
              if (element) rows.current.set(item.id, element);
              else rows.current.delete(item.id);
            }}
            tabIndex={item.id === current ? 0 : -1} selected={item.id === selected}
            draggable={!!dragType} onDragStart={dragType ? (event) => event.dataTransfer.setData(dragType, item.id) : undefined}
            onFocus={() => setFocused(item.id)}
            onClick={() => onInspect(item.id)}
            onDoubleClick={onPlace && (() => onPlace(item.id))}
            onKeyDown={(event) => onKeyDown(event, item.id)} />)}
        </div>
      </DrawerSection>)}
    </StudioRegion>
  </DrawerSection>;
}
