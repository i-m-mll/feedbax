import { Search } from 'lucide-react';
import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { HoverCard, HoverCardTrigger, createHoverCardHandle } from '@/components/ui/HoverCard';
import { CarryChip, carry, type CarryKind } from '@/components/ui/carry';
import { ListRow } from '@/components/ui/ListRow';
import { popupSurface } from '@/components/ui/styles';
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
  /** What the item's hover card says about it. */
  about?: LibraryAbout;
  /** Why the item cannot be placed; such an item is dimmed, its hover card says why, and it places nothing. */
  refusal?: string;
}

export interface LibraryAbout {
  description: string;
  inputs: readonly string[];
  outputs: readonly string[];
  parameters: readonly string[];
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
  /** What a drag of an item carries it as; without it, items are not dragged. */
  carried?: CarryKind<LibraryItem>;
}

function AboutCard({ item: { name, about, refusal } }: { item: LibraryItem & { about: LibraryAbout } }) {
  const lists = (['inputs', 'outputs', 'parameters'] as const).filter((list) => about[list].length);
  return <div className="grid gap-2">
    <div className="font-medium">{name}</div>
    {refusal && <p role="status">{refusal}</p>}
    {about.description && <p className="text-shell-muted">{about.description}</p>}
    {lists.length > 0 && <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
      {lists.map((list) => <div key={list} className="contents">
        <dt className="text-shell-muted">{studioName(list)}</dt>
        <dd className="min-w-0 break-words font-mono text-sm">{about[list].join(', ')}</dd>
      </div>)}
    </dl>}
  </div>;
}

function matches(item: LibraryItem, query: string) {
  return `${item.name} ${item.group} ${item.terms ?? ''}`.toLowerCase().includes(query);
}

/** A drawer section holding a catalog: a search field over groups of rows, folded until opened or searched. */
export function LibraryList({
  state, items, sections, section, name, searchName, selected, onInspect, onPlace, carried,
}: LibraryListProps) {
  const [query, setQuery] = useState('');
  const names = useId();
  const [focused, setFocused] = useState<string>();
  const [card] = useState(() => createHoverCardHandle<LibraryItem & { about: LibraryAbout }>());
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
    } else if (event.key === 'Enter' && onPlace && !shown[at]?.refusal) {
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
          {list.map((item) => {
            const placeable = !item.refusal;
            const row = <ListRow key={item.id} role="option" name={item.name} glyph={item.glyph}
              disabled={!placeable} aria-description={item.refusal}
              ref={(element) => {
                if (element) rows.current.set(item.id, element);
                else rows.current.delete(item.id);
              }}
              tabIndex={item.id === current ? 0 : -1} selected={item.id === selected}
              onPointerDown={carried && placeable ? (event) => carry(event, carried, item) : undefined}
              onFocus={() => setFocused(item.id)}
              onClick={() => onInspect(item.id)}
              onDoubleClick={onPlace && placeable ? () => onPlace(item.id) : undefined}
              onKeyDown={(event) => onKeyDown(event, item.id)} />;
            const { about } = item;
            return about ? <HoverCardTrigger key={item.id} handle={card} payload={{ ...item, about }} render={row} />
              : row;
          })}
        </div>
      </DrawerSection>)}
    </StudioRegion>
    <CarryChip kind={carried}>
      {(item) => <ListRow name={item.name} glyph={item.glyph} className={popupSurface} />}
    </CarryChip>
    <HoverCard handle={card} beside={(trigger) => trigger.closest('[data-section-stack]')}>
      {(item) => <AboutCard item={item} />}
    </HoverCard>
  </DrawerSection>;
}
