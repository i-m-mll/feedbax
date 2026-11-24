import { appTip } from '@window-shell/browser/tooltip';
import { ChevronDown, FileCode } from 'lucide-react';
import { useRef } from 'react';
import { studioAccessibleName, studioCountOf, studioName, type StudioNameId } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { InlineEdit } from '@/components/ui/InlineEdit';
import {
  Menu, MenuContent, MenuItem, MenuRadioGroup, MenuRadioItem, MenuSeparator, MenuSubmenu, MenuTrigger,
} from '@/components/ui/Menu';
import { StatusMark } from '@/components/ui/StatusMark';
import { isMemberName } from '@/features/projectFolder/projectMemberName';

/** What the slot header can do with the document its view shows. An absent command is not offered. */
export interface DocumentPickerModel {
  readonly label: StudioNameId;
  readonly document: string;
  /** The shown document's label, or its key. */
  readonly name: string;
  /** The documents the step can read, each after the document it is based on, at its depth below its root. */
  readonly documents: readonly PickerDocument[];
  /** What New can start from, when it offers more than an empty document; `create` takes the chosen one's key. */
  readonly templates?: readonly PickerTemplate[];
  readonly count?: { readonly lit: number; readonly users: number };
  /** Why the last document chosen, or the last change to the shown one, was not taken. */
  readonly refusal?: string;
  readonly choose?: (document: string) => void;
  /** Each resolves to the document now shown, whose name then goes into editing. */
  readonly create?: (template?: string) => Promise<string | undefined>;
  readonly duplicate?: () => Promise<string | undefined>;
  /** Sets the document's label; its key is the clearing value. */
  readonly rename?: (label: string | null) => void;
  readonly changeKey?: (key: string) => void;
  readonly detach?: () => void;
}

export interface PickerDocument {
  readonly key: string;
  readonly name: string;
  readonly depth: number;
  /** The document is defined in Python by an installed package, not written in Studio. */
  readonly python?: boolean;
  /** Why the step cannot use the document as it stands, such as a model input it would leave unfed. */
  readonly problem?: string;
}

export interface PickerTemplate {
  readonly key: string;
  readonly name: string;
  readonly python?: boolean;
}

function PythonMark() {
  return <span role="img" aria-label={studioName('python')} {...appTip(studioName('python'))}
    className="flex shrink-0 text-shell-muted [&_svg]:size-[1em]"><FileCode aria-hidden="true" /></span>;
}

/** A document's marks: Python, and the problem that keeps the step from using it, whose tip is the reason. */
function DocumentMarks({ python, problem }: { python?: boolean; problem?: string }) {
  return <>
    {python && <PythonMark />}
    {problem && <StatusMark status="cannot_run" label={problem} />}
  </>;
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
  const { label, document, name, documents, templates, count, refusal, choose, create, duplicate, rename, changeKey, detach } = model;
  const shown = documents.find((entry) => entry.key === document);
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
      {shown && <DocumentMarks python={shown.python} problem={shown.problem} />}
      {count && count.users > 1 && count.lit < count.users &&
        <span className="whitespace-nowrap text-xs text-shell-muted">{studioCountOf(count.lit, count.users)}</span>}
      <ChevronDown aria-hidden="true" />
    </MenuTrigger>
    <MenuContent aria-label={studioName(label)} finalFocus={() => !renaming.current}>
      <MenuRadioGroup value={document} onValueChange={(value: string) => { if (value !== document) choose?.(value); }}>
        {documents.map(({ key, name, depth, python, problem }) => <MenuRadioItem key={key} value={key} disabled={!choose}
          mark={<DocumentMarks python={python} problem={problem} />}>
          <span style={depth ? { paddingInlineStart: `${depth}rem` } : undefined}>{name}</span>
        </MenuRadioItem>)}
      </MenuRadioGroup>
      <MenuSeparator />
      {templates && create
        ? <MenuSubmenu label={studioName('new')}>
          <MenuItem onClick={edit(() => create())}>{studioName('blank')}</MenuItem>
          {templates.map((template) => <MenuItem key={template.key} onClick={edit(() => create(template.key))}>
            {template.name}{template.python && <PythonMark />}
          </MenuItem>)}
        </MenuSubmenu>
        : <MenuItem disabled={!create} onClick={edit(create && (() => create()))}>{studioName('new')}</MenuItem>}
      <MenuItem disabled={!duplicate} onClick={edit(duplicate)}>{studioName('duplicate')}</MenuItem>
      <MenuItem disabled={!rename} onClick={start('name')}>{studioName('rename')}</MenuItem>
      <MenuItem disabled={!changeKey} onClick={start('key')}>{studioName('changeKey')}</MenuItem>
      {detach && <MenuItem onClick={detach}>{studioName('detach')}</MenuItem>}
    </MenuContent>
  </Menu>
  {refusal && <span role="alert" className="min-w-0 truncate text-xs text-shell-muted">{refusal}</span>}</>;
}
