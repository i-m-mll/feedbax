import { ChevronDown } from 'lucide-react';
import { studioAccessibleName, studioCountOf, studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { InlineEdit } from '@/components/ui/InlineEdit';
import {
  Menu, MenuContent, MenuItem, MenuRadioGroup, MenuRadioItem, MenuSeparator, MenuTrigger,
} from '@/components/ui/Menu';
import { isMemberName } from '@/features/projectFolder/projectMemberName';

/** What the slot header can do with the document its view shows. An absent command is not offered. */
export interface DocumentPickerModel {
  readonly label: StudioNameId;
  readonly document: string;
  readonly documents: readonly string[];
  readonly count?: { readonly lit: number; readonly users: number };
  readonly choose?: (document: string) => void;
  /** Each resolves to the document now shown, whose name then goes into editing. */
  readonly create?: () => Promise<string | undefined>;
  readonly duplicate?: () => Promise<string | undefined>;
  readonly rename?: (name: string) => void;
}

/**
 * `editing` names the document whose name is being edited. Its owner outlives the picker, because the view
 * can stop drawing its header while the document it has just created or copied loads.
 */
export function DocumentPicker({ model, editing, setEditing }: {
  model: DocumentPickerModel;
  editing: string | undefined;
  setEditing: (document: string | undefined) => void;
}) {
  const { label, document, documents, count, choose, create, duplicate, rename } = model;
  const edit = (start?: () => Promise<string | undefined>) => start && (async () => setEditing(await start()));

  if (editing === document && rename) {
    return <InlineEdit value={document} editing className="studio-window-slot__document"
      aria-label={studioAccessibleName('rename', document)}
      onEditingChange={(next) => { if (!next) setEditing(undefined); }}
      onCommit={(name) => {
        if (!isMemberName(name) || documents.includes(name)) return false;
        rename(name);
      }} />;
  }
  return <Menu>
    <MenuTrigger render={<Button variant="ghost" className="min-w-0 gap-1 px-1" />}
      aria-label={studioAccessibleName(label, document)}>
      <span className="studio-window-slot__document">{document}</span>
      {count && count.users > 1 && count.lit < count.users &&
        <span className="whitespace-nowrap text-xs text-shell-muted">{studioCountOf(count.lit, count.users)}</span>}
      <ChevronDown aria-hidden="true" />
    </MenuTrigger>
    <MenuContent aria-label={studioName(label)}>
      <MenuRadioGroup value={document} onValueChange={(value: string) => { if (value !== document) choose?.(value); }}>
        {documents.map((name) => <MenuRadioItem key={name} value={name} disabled={!choose}>{name}</MenuRadioItem>)}
      </MenuRadioGroup>
      <MenuSeparator />
      <MenuItem disabled={!create} onClick={edit(create)}>{studioName('new')}</MenuItem>
      <MenuItem disabled={!duplicate} onClick={edit(duplicate)}>{studioName('duplicate')}</MenuItem>
      <MenuItem disabled={!rename} onClick={() => setEditing(document)}>{studioName('rename')}</MenuItem>
    </MenuContent>
  </Menu>;
}
