import type { StudioDensity } from '@/generated/studioContracts';
import { studioFieldLabel, studioName } from '@/components/activity/studioVocabulary';
import { centred, drawFamilies, drawText, perIsFree, withFamily, type DrawDraft, type FieldModel } from './fieldModel';
import { partDraft, setPartDraft } from './partDrafts';
import { ValueField } from './ValueField';

/**
 * Starts a draw in the field around `centre`: of the site's first family that has a centre, with `centre` as it, or
 * else of its first family with its parts empty.
 */
export function startDraw(model: FieldModel, centre?: string) {
  const families = drawFamilies(model);
  const family = families.find(centred) ?? families[0];
  if (!family) return;
  const draw = withFamily(model, undefined, family);
  setPartDraft(model.id, { kind: 'draw', draw: { ...draw, args: draw.args.map((arg, index) => index === 0 && centred(family) && centre?.trim() ? centre : arg) } });
}

/**
 * A draw's parts under its field: the family, each parameter as a field, Per where the site leaves it free, and the
 * density strip. An edit of a part writes the draw's text once every part is filled.
 */
export function DrawParts({ model, draw }: { model: FieldModel; draw: DrawDraft }) {
  const families = drawFamilies(model);
  const family = families.find(({ id }) => id === draw.family);
  const pending = partDraft(model.id)?.kind === 'draw';
  const commit = model.commit;
  async function write(next: DrawDraft) {
    const text = drawText(next);
    if (!text || !commit) {
      setPartDraft(model.id, { kind: 'draw', draw: next });
      return undefined;
    }
    const failure = await commit({ kind: 'text', text });
    setPartDraft(model.id, failure ? { kind: 'draw', draw: next } : undefined);
    return failure;
  }
  const base = { names: model.names, functions: model.functions, kinds: [], kind: 'value' } as const;
  const labels = family?.parameters?.length
    ? family.parameters.map(({ label }) => label)
    : draw.args.map((_, index) => String(index + 1));
  const parts: FieldModel[] = [
    {
      ...base, id: model.id + '.family', label: studioName('family'), text: null, value: draw.family,
      type: { kind: 'choice', options: families.map(({ id, label }) => ({ value: id, label: label ?? id })) },
      commit: commit && ((edit) => {
        const chosen = edit.kind === 'set' && families.find(({ id }) => id === edit.value);
        return chosen && chosen.id !== draw.family ? write(withFamily(model, draw, chosen)) : Promise.resolve(undefined);
      }),
    },
    ...draw.args.map((arg, index): FieldModel => ({
      ...base, id: `${model.id}.${index}`, label: labels[index], text: arg ?? '', type: { kind: 'number' },
      commit: commit && ((edit) => edit.kind === 'text'
        ? write({ ...draw, args: draw.args.map((value, at) => at === index ? edit.text : value) })
        : Promise.resolve(undefined)),
    })),
    ...perIsFree(model) ? [{
      ...base, id: model.id + '.per', label: studioName('per'), text: null, value: draw.per,
      type: { kind: 'choice', options: [
        { value: 'trial', label: studioName('trial') }, { value: 'run', label: studioName('run') },
      ] },
      commit: commit && ((edit: Parameters<NonNullable<FieldModel['commit']>>[0]) =>
        edit.kind === 'set' && edit.value !== draw.per
          ? write({ ...draw, per: edit.value === 'run' ? 'run' : 'trial' }) : Promise.resolve(undefined)),
    } satisfies FieldModel] : [],
  ];
  const density = pending ? undefined : model.structure?.density;
  return <div className="grid gap-1">
    {parts.map((part) => <ValueField key={part.id + JSON.stringify(part.value ?? part.text)} model={part} />)}
    {density && <DensityStrip density={density} label={model.label} />}
  </div>;
}

function DensityStrip({ density, label }: { density: StudioDensity; label: string }) {
  const { values } = density;
  const step = 100 / Math.max(values.length - 1, 1);
  const line = values.map((value, index) => `${index ? 'L' : 'M'}${(index * step).toFixed(2)},${(16 - value * 14).toFixed(2)}`).join(' ');
  return <svg role="img" aria-label={studioFieldLabel('density', label)} viewBox="0 0 100 16" preserveAspectRatio="none"
    className="value-field-density h-4 w-full text-shell-muted">
    <path d={`${line} L100,16 L0,16 Z`} fill="currentColor" fillOpacity={0.2} stroke="none" />
    <path d={line} fill="none" stroke="currentColor" strokeWidth={1} vectorEffect="non-scaling-stroke" />
  </svg>;
}
