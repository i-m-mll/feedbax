import { useState } from 'react';
import { studioName, studioParameterLabel } from '@/components/activity/studioVocabulary';
import { FieldProblem } from '@/components/ui/FieldProblem';
import { FieldRow } from '@/components/ui/FieldRow';
import { TreeChoice } from '@/components/ui/TreeChoice';
import { linkNames, nameText, type FieldModel } from './fieldModel';
import { setPartDraft } from './partDrafts';
import { usePartMenu } from './partMenu';

/**
 * A Link's part: the name it reads, a choice over the names the site can read, by their paths where they have them.
 * `opened` draws the list open, for a Link just chosen as the field's kind.
 */
export function LinkParts({ model, reference, opened }: { model: FieldModel; reference?: string; opened: boolean }) {
  const [open, setOpen] = useState(opened);
  const menu = usePartMenu(model, [], open);
  const names = linkNames(model, reference, menu);
  const commit = model.commit;
  return <FieldRow label={studioName('name')}>
    <TreeChoice aria-label={studioParameterLabel(model.label, studioName('name'))} className="w-full" disabled={!commit} defaultOpen={opened} value={reference ?? null}
      onOpenChange={setOpen} invalid={menu.kind === 'failed'}
      trailing={menu.kind === 'failed' && <FieldProblem reason={menu.reason} />}
      options={names.map(({ id, name }) => {
        const path = model.namePaths?.[id] ?? [];
        return { value: id, path: path.at(-1) === name ? path : [...path, name] };
      })}
      onValueChange={async (id) => {
        const chosen = names.find((name) => name.id === id);
        if (!chosen || !commit || chosen.id === reference) return;
        if (!await commit({ kind: 'text', text: nameText(chosen.name) })) setPartDraft(model.id, undefined);
      }} />
  </FieldRow>;
}
