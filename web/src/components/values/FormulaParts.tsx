import type { FunctionInfo } from '@/generated/studioContracts';
import { studioName } from '@/components/activity/studioVocabulary';
import { callText, parameterNames, rootFunctions, withFunction, type CallDraft, type FieldEdit, type FieldModel } from './fieldModel';
import { setPartDraft } from './partDrafts';
import { ValueField } from './ValueField';

/**
 * A call's parts under its field: its function as a choice among `offered`, and each positional argument as a field.
 * Choosing a function keeps the arguments; an edit of a part writes the call's text once every argument is filled.
 */
export function CallParts({ model, call, offered, kind }: {
  model: FieldModel; call: CallDraft; offered: readonly FunctionInfo[]; kind: 'formula' | 'shape';
}) {
  const commit = model.commit;
  async function write(next: CallDraft) {
    const text = callText(model, next);
    if (!text || !commit) {
      setPartDraft(model.id, { kind, call: next });
      return undefined;
    }
    if (text === model.text) {
      setPartDraft(model.id, undefined);
      return undefined;
    }
    const failure = await commit({ kind: 'text', text });
    setPartDraft(model.id, failure ? { kind, call: next } : undefined);
    return failure;
  }
  const current = model.functions.find(({ id }) => id === call.function);
  const labels = current ? parameterNames(current) : [];
  const base = { names: model.names, functions: model.functions, kinds: [], kind: 'value' } as const;
  const parts: FieldModel[] = [
    {
      ...base, id: model.id + '.function', label: studioName('function'), text: null, value: call.function,
      type: { kind: 'choice', options: offered.map(({ id, label }) => ({ value: id, label: label ?? id })) },
      commit: commit && ((edit: FieldEdit) => {
        const chosen = edit.kind === 'set' && offered.find(({ id }) => id === edit.value);
        return chosen && chosen.id !== call.function ? write(withFunction(call, chosen)) : Promise.resolve(undefined);
      }),
    },
    ...call.args.map((arg, index): FieldModel => ({
      ...base, id: `${model.id}.${index}`, label: labels[index] ?? String(index + 1), text: arg ?? '', type: { kind: 'string' },
      commit: commit && ((edit) => edit.kind === 'text'
        ? write({ ...call, args: call.args.map((value, at) => at === index ? edit.text : value) })
        : Promise.resolve(undefined)),
    })),
  ];
  return <div className="grid gap-1">
    {parts.map((part) => <ValueField key={part.id + JSON.stringify(part.value ?? part.text)} model={part} />)}
  </div>;
}

/** A Formula's parts: its outer function, chosen among those that fit at its root, and its arguments. */
export function FormulaParts({ model, call }: { model: FieldModel; call: CallDraft }) {
  return <CallParts model={model} call={call} offered={rootFunctions(model, call.function)} kind="formula" />;
}
