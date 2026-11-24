import { studioFieldLabel } from '@/components/activity/studioVocabulary';
import { shapeFunctions, type CallDraft, type FieldModel } from './fieldModel';
import { CallParts } from './FormulaParts';
import { partDraft } from './partDrafts';

/** A Shape's parts: its function among the site's shapes, its parameters, and its curve over the window. */
export function ShapeParts({ model, call }: { model: FieldModel; call: CallDraft }) {
  const pending = partDraft(model.id)?.kind === 'shape';
  return <div className="grid gap-1">
    <CallParts model={model} call={call} offered={shapeFunctions(model)} kind="shape" />
    {!pending && model.curve && <Curve curve={model.curve} label={model.label} />}
  </div>;
}

/** The value over the steps where it is read, scaled to the strip's height; a step where it is not read is a gap. */
function Curve({ curve, label }: { curve: readonly (number | null)[]; label: string }) {
  const read = curve.flatMap((value, step) => value === null ? [] : [step]);
  if (!read.length) return null;
  const first = read[0];
  const span = Math.max(read.at(-1)! - first, 1);
  const values = read.map((step) => curve[step]!);
  const low = Math.min(...values);
  const range = Math.max(...values) - low || 1;
  let line = '';
  curve.forEach((value, step) => {
    if (value === null) return;
    const x = ((step - first) / span * 100).toFixed(2);
    const y = (15 - (value - low) / range * 14).toFixed(2);
    line += `${step > first && curve[step - 1] !== null ? 'L' : 'M'}${x},${y} `;
  });
  return <svg role="img" aria-label={studioFieldLabel('curve', label)} viewBox="0 0 100 16" preserveAspectRatio="none"
    className="h-8 w-full text-shell-muted">
    <path d={line} fill="none" stroke="currentColor" strokeWidth={1} vectorEffect="non-scaling-stroke" />
  </svg>;
}
