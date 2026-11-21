import { ChevronDown } from 'lucide-react';
import { useRef } from 'react';
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
  /** The shown document's label, or its key. */
  readonly name: string;
  /** The documents the step can read, each after the document it is based on, at its depth below its root. */
  readonly documents: readonly { readonly key: string; readonly name: string; readonly depth: number }[];
  readonly count?: { readonly lit: number; readonly users: number };
  /** Why the last document chosen, or the last change to the shown one, was not taken. */
  readonly refusal?: string;
  readonly choose?: (document: string) => void;
  /** Each resolves to the document now shown, whose name then goes into editing. */
  readonly create?: () => Promise<string | undefined>;
  readonly duplicate?: () => Promise<string | undefined>;
  /** Sets the document's label; its key is the clearing value. */
  readonly rename?: (label: string | null) => void;
  readonly changeKey?: (key: string) => void;
  readonly detach?: () => void;
}

/** The document whose name or key the slot header is editing. */
export interface DocumentEditing {
  readonly document: string;
  readonly part: 'name' | 'key';
}

/**
 * `editing` names the document whose name or key is being edited. Its owner outlives the picker, because the view
 * can stop drawing its header while the document it has just created or copied loads.
 */
export function DocumentPicker({ model, editing, setEditing }: {
  model: DocumentPickerModel;
  editing: DocumentEditing | undefined;
  setEditing: (editing: DocumentEditing | undefined) => void;
}) {
  const { label, document, name, documents, count, refusal, choose, create, duplicate, rename, changeKey, detach } = model;
  const edit = (start?: () => Promise<string | undefined>) => start && (async () => {
    const created = await start();
    setEditing(created === undefined ? undefined : { document: created, part: 'name' });
  });
  // Rename and Change Key replace the menu's trigger with a field, so the closing menu must not return focus elsewhere.
  const renaming = useRef(false);
  const field = editing?.document === document ? editing.part : undefined;
  const close = (next: boolean) => { if (!next) setEditing(undefined); };

  if (field === 'name' && rename) {
    return <InlineEdit value={name} editing className="studio-window-slot__document"
      aria-label={studioAccessibleName('rename', name)} onEditingChange={close}
      onCommit={(next) => rename(next === document ? null : next)} />;
  }
  if (field === 'key' && changeKey) {
    return <InlineEdit value={document} editing className="studio-window-slot__document font-mono"
      aria-label={studioAccessibleName('changeKey', name)} onEditingChange={close}
      onCommit={(key) => {
        if (!isMemberName(key) || documents.some((entry) => entry.key === key)) return false;
        changeKey(key);
      }} />;
  }
  const start = (part: DocumentEditing['part']) => () => { renaming.current = true; setEditing({ document, part }); };
  return <><Menu onOpenChange={(open) => { if (open) renaming.current = false; }}>
    <MenuTrigger render={<Button variant="ghost" className="min-w-0 gap-1 px-1" />}
      aria-label={studioAccessibleName(label, name)}>
      <span className="studio-window-slot__document">{name}</span>
      {count && count.users > 1 && count.lit < count.users &&
        <span className="whitespace-nowrap text-xs text-shell-muted">{studioCountOf(count.lit, count.users)}</span>}
      <ChevronDown aria-hidden="true" />
    </MenuTrigger>
    <MenuContent aria-label={studioName(label)} finalFocus={() => !renaming.current}>
      <MenuRadioGroup value={document} onValueChange={(value: string) => { if (value !== document) choose?.(value); }}>
        {documents.map(({ key, name, depth }) => <MenuRadioItem key={key} value={key} disabled={!choose}>
          <span style={depth ? { paddingInlineStart: `${depth}rem` } : undefined}>{name}</span>
        </MenuRadioItem>)}
      </MenuRadioGroup>
      <MenuSeparator />
      <MenuItem disabled={!create} onClick={edit(create)}>{studioName('new')}</MenuItem>
      <MenuItem disabled={!duplicate} onClick={edit(duplicate)}>{studioName('duplicate')}</MenuItem>
      <MenuItem disabled={!rename} onClick={start('name')}>{studioName('rename')}</MenuItem>
      <MenuItem disabled={!changeKey} onClick={start('key')}>{studioName('changeKey')}</MenuItem>
      {detach && <MenuItem onClick={detach}>{studioName('detach')}</MenuItem>}
    </MenuContent>
  </Menu>
  {refusal && <span role="alert" className="min-w-0 truncate text-xs text-shell-muted">{refusal}</span>}</>;
}
