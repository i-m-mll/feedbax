import { MoreHorizontal } from 'lucide-react';
import { useState } from 'react';
import { studioAccessibleName, studioFormat, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { InlineEdit } from '@/components/ui/InlineEdit';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Section } from '@/components/ui/Section';
import type { FieldModel } from '@/components/values/fieldModel';
import { ValueField } from '@/components/values/ValueField';

/** One value of a named-value rule: the Service's name for it, the field of its value, and how to rename or remove it. */
export interface FamilyValue {
  readonly name: string;
  readonly field: FieldModel;
  readonly rename?: (name: string) => void;
  readonly remove?: () => void;
}

/**
 * One index of a family: its name as authored, and either the field of its text rule or the values of its named-value
 * rule, as the Service names them.
 */
export interface FamilyIndex {
  readonly name: string;
  readonly rule?: FieldModel;
  readonly values?: readonly FamilyValue[];
  readonly remove?: () => void;
}

/** The muted count after a family member's name: how many members it generates, or nothing while its rule has a Problem. */
export function FamilyCount({ count }: { count: number | null | undefined }) {
  return count == null ? null : <span className="shrink-0 text-shell-muted tabular-nums">{studioFormat('memberCount', { count })}</span>;
}

/** A family member's name with its count, where it has one, as text: `Initial position {axis} {sign} × 4`. */
export function familyText(name: string, count: number | null | undefined) {
  return count == null ? name : `${name} ${studioFormat('memberCount', { count })}`;
}

/** The removals of an index or of a named value, in a menu named for what they remove from. */
function IndexMenu({ of, items }: { of: string; items: readonly { name: string; remove: () => void }[] }) {
  return <Menu>
    <MenuTrigger render={<IconButton variant="ghost" size="mark" label={studioAccessibleName('remove', of)}
      icon={<MoreHorizontal aria-hidden="true" />} />} />
    <MenuContent aria-label={studioAccessibleName('remove', of)}>
      {items.map(({ name, remove }) => <MenuItem key={name} onClick={remove}>{studioAccessibleName('remove', name)}</MenuItem>)}
    </MenuContent>
  </Menu>;
}

function NamedValue({ value }: { value: FamilyValue }) {
  const [editing, setEditing] = useState(false);
  return <div className="grid gap-0.5">
    {value.rename ? <InlineEdit form="cell" value={value.name} editing={editing} onEditingChange={setEditing}
      aria-label={studioAccessibleName('rename', value.name)}
      className="min-w-0 text-xs text-shell-muted"
      onCommit={(next) => { if (next.trim() && next.trim() !== value.name) value.rename!(next.trim()); }} />
      : <span className="text-xs text-shell-muted">{value.name}</span>}
    <ValueField key={value.name + JSON.stringify(value.field.value ?? value.field.text)} model={value.field} />
  </div>;
}

/** A family's Over group: each index in `over` order, then how they combine when there are two or more. */
export function FamilyOver({ indices, combination }: { indices: readonly FamilyIndex[]; combination?: FieldModel }) {
  const removable = indices.flatMap(({ name, remove }) => remove ? [{ name, remove }] : []);
  return <Section level="group" name={studioName('over')} bodyClassName="grid gap-1"
    actions={removable.length > 0 && <IndexMenu of={studioName('over')} items={removable} />}>
    {indices.map(({ name, rule, values }) => {
      if (rule) return <ValueField key={name + JSON.stringify(rule.value ?? rule.text)} model={rule} />;
      const removableValues = (values ?? []).flatMap((value) => value.remove ? [{ name: value.name, remove: value.remove }] : []);
      return <FieldRow key={name} label={name}
        actions={removableValues.length > 0 && <IndexMenu of={name} items={removableValues} />}>
        {values && <div className="grid gap-1">{values.map((value) => <NamedValue key={value.name} value={value} />)}</div>}
      </FieldRow>;
    })}
    {indices.length > 1 && combination && <ValueField model={combination} />}
  </Section>;
}
