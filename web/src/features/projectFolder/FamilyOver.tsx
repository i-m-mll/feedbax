import { MoreHorizontal } from 'lucide-react';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { FieldRow } from '@/components/ui/FieldRow';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Section } from '@/components/ui/Section';
import type { FieldModel } from '@/components/values/fieldModel';
import { ValueField } from '@/components/values/ValueField';

/**
 * One index of a family: its name as authored, and the field of its rule when the rule is text. A rule of named
 * values has no field here until the Service names its values.
 */
export interface FamilyIndex {
  readonly name: string;
  readonly rule?: FieldModel;
  readonly remove?: () => void;
}

/** A family's Over group: each index in `over` order, then how they combine when there are two or more. */
export function FamilyOver({ indices, combination }: { indices: readonly FamilyIndex[]; combination?: FieldModel }) {
  const removable = indices.filter(({ remove }) => remove);
  return <Section level="group" name={studioName('over')} bodyClassName="grid gap-1"
    actions={removable.length > 0 && <Menu>
      <MenuTrigger render={<IconButton variant="ghost" size="mark" label={studioAccessibleName('over', studioName('remove'))}
        icon={<MoreHorizontal aria-hidden="true" />} />} />
      <MenuContent aria-label={studioAccessibleName('over', studioName('remove'))}>
        {removable.map(({ name, remove }) =>
          <MenuItem key={name} onClick={remove}>{studioAccessibleName('remove', name)}</MenuItem>)}
      </MenuContent>
    </Menu>}>
    {indices.map(({ name, rule }) => rule
      ? <ValueField key={name + JSON.stringify(rule.value ?? rule.text)} model={rule} />
      : <FieldRow key={name} label={name} />)}
    {indices.length > 1 && combination && <ValueField model={combination} />}
  </Section>;
}
