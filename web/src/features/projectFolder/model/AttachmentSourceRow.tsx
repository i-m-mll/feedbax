import { Crosshair, Variable, X } from 'lucide-react';
import { studioAccessibleName } from '@/components/activity/studioVocabulary';
import { IconButton } from '@/components/ui/Button';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import type { FieldModel } from '@/components/values/fieldModel';
import { ValueField } from '@/components/values/ValueField';

/**
 * A value an attachment reads, from one of three sources: its own number or numbers (`field`), a Task variable chosen
 * from `variables`, or a port's signal, picked on the canvas by `pick`. `linked` names the variable or port it reads,
 * and `clear` gives it back its own value.
 */
export function AttachmentSourceRow({ label, field, linked, variables, choose, pick, clear }: {
  label: string;
  field: FieldModel;
  linked?: string;
  variables: readonly { readonly id: string; readonly name: string }[];
  choose?: (variable: string) => void;
  pick?: () => void;
  clear?: () => void;
}) {
  return <div className="flex min-w-0 items-start gap-1">
    <div className="min-w-0 flex-1"><ValueField model={field} /></div>
    {choose && variables.length > 0 && <Menu>
      <MenuTrigger render={<IconButton label={studioAccessibleName('link', label)} icon={<Variable />} />} />
      <MenuContent aria-label={studioAccessibleName('link', label)}>
        {variables.map(({ id, name }) => <MenuItem key={id} onClick={() => choose(id)}>{name}</MenuItem>)}
      </MenuContent>
    </Menu>}
    {pick && <IconButton label={studioAccessibleName('pick', label)} icon={<Crosshair />} onClick={pick} />}
    {linked !== undefined && clear && <IconButton label={studioAccessibleName('clearLink', label)} icon={<X />} onClick={clear} />}
  </div>;
}
