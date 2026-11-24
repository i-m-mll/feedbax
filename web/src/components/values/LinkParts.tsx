import { useId } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { Combobox } from '@/components/ui/Combobox';
import { FieldRow } from '@/components/ui/FieldRow';
import { linkNames, nameText, type FieldModel } from './fieldModel';
import { setPartDraft } from './partDrafts';

/**
 * A Link's part: the name it reads, a choice over the names the site can read, each shown with its path where it has
 * one. `opened` draws the list open, for a Link just chosen as the field's kind.
 */
export function LinkParts({ model, reference, opened }: { model: FieldModel; reference?: string; opened: boolean }) {
  const labelId = useId();
  const names = linkNames(model);
  const commit = model.commit;
  return <FieldRow label={studioName('name')} labelId={labelId}>
    <Combobox aria-labelledby={labelId} className="w-full" disabled={!commit} defaultOpen={opened}
      value={reference ?? null} placeholder={studioName('name')}
      items={names.map(({ id, name }) => {
        const path = model.namePaths?.[id] ?? [];
        const levels = path.at(-1) === name ? path.slice(0, -1) : path;
        return { value: id, label: name, detail: levels.length ? levels.join(' / ') : undefined };
      })}
      onValueChange={async (id) => {
        const chosen = names.find((name) => name.id === id);
        if (!chosen || !commit || chosen.id === reference) return;
        if (!await commit({ kind: 'text', text: nameText(chosen.name) })) setPartDraft(model.id, undefined);
      }} />
  </FieldRow>;
}
