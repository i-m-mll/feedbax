import { useState, type FormEvent } from "react";
import { studioAccessibleName, studioCountOf, studioName } from "@/components/activity/studioVocabulary";
import type { PlanInputView, PlanSweepView } from "@/features/pipeline/planView";
import type { SweepSelectionTerm } from "@/features/pipeline/selection";

interface InputChoice {
  readonly id: string;
  readonly label: string;
  readonly kind: "step" | "run" | "subset";
}

interface MeasureRule {
  readonly metricId: string;
  readonly order: "min" | "max";
  readonly count: number;
  readonly per: readonly string[];
}

interface InputSubsetView {
  readonly sweeps: readonly PlanSweepView[];
  readonly rows: readonly { id: string; label: string }[];
  readonly metrics: readonly { id: string; label: string }[];
  readonly values: readonly SweepSelectionTerm[];
  readonly rowIds: readonly string[];
  readonly measure?: MeasureRule;
  readonly members: readonly ({ readonly id: string; readonly label: string } | { readonly rank: number })[];
  readonly name?: string;
}

export interface StudioRunInputView extends PlanInputView {
  readonly choices: readonly InputChoice[];
  readonly choiceId?: string;
  readonly reason?: string;
  readonly editor?: InputSubsetView;
  readonly actions: readonly StudioInputEdit["kind"][];
}

export type StudioInputEdit =
  | { readonly kind: "choose"; readonly choiceId: string }
  | { readonly kind: "value"; readonly sweepId: string; readonly valueId: string; readonly included: boolean }
  | { readonly kind: "row"; readonly rowId: string; readonly included: boolean }
  | { readonly kind: "measure"; readonly rule: MeasureRule | null }
  | { readonly kind: "saveSubset" | "rename"; readonly name: string }
  | { readonly kind: "toRows" };

const inputClass = "rounded border border-shell-control bg-shell-surface px-2 py-1 disabled:opacity-50";

function MeasureEditor({ editor, onEdit }: { editor: InputSubsetView; onEdit: (rule: MeasureRule | null) => void }) {
  const measure = editor.measure;
  return (
    <fieldset className="flex flex-wrap items-baseline gap-2">
      <legend className="font-medium">{studioName("metric")}</legend>
      <select
        aria-label={studioName("metric")} className={inputClass} value={measure?.metricId ?? ""}
        onChange={(event) => {
          const metricId = event.currentTarget.value;
          onEdit(metricId ? { ...measure, metricId, order: measure?.order ?? "min", count: measure?.count ?? 1, per: measure?.per ?? [] } : null);
        }}
      >
        <option value="">{studioName("choose")}</option>
        {editor.metrics.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
      </select>
      {measure && (
        <>
          <select aria-label={studioName("order")} value={measure.order} className={inputClass} onChange={(event) => onEdit({ ...measure, order: event.currentTarget.value === "max" ? "max" : "min" })}>
            <option value="min">{studioName("ascending")}</option>
            <option value="max">{studioName("descending")}</option>
          </select>
          <input
            aria-label={studioName("count")} className={`${inputClass} w-24`} type="number" min={1} step={1}
            key={measure.count} defaultValue={measure.count}
            onBlur={({ currentTarget }) => {
              const count = currentTarget.valueAsNumber;
              if (currentTarget.validity.valid && Number.isInteger(count) && count !== measure.count) onEdit({ ...measure, count });
            }}
          />
          <fieldset className="flex gap-2">
            <legend>{studioName("per")}</legend>
            {editor.sweeps.map((sweep) => (
              <label key={sweep.id} className="flex gap-1">
                <input type="checkbox" checked={measure.per.includes(sweep.id)} onChange={(event) => onEdit({
                  ...measure,
                  per: event.currentTarget.checked ? [...measure.per, sweep.id] : measure.per.filter((id) => id !== sweep.id),
                })} />
                {sweep.label}
              </label>
            ))}
          </fieldset>
        </>
      )}
    </fieldset>
  );
}

function SubsetNameForm({ name, onEdit }: { name?: string; onEdit: (edit: StudioInputEdit) => void }) {
  const [draft, setDraft] = useState(name ?? "");
  const action = name ? "rename" : "saveSubset";
  function submit(event: FormEvent) {
    event.preventDefault();
    const value = draft.trim();
    if (value && value !== name) onEdit({ kind: action, name: value });
  }
  return (
    <form onSubmit={submit} className="flex flex-wrap gap-1">
      <input aria-label={studioName("name")} className={inputClass} value={draft} onChange={(event) => setDraft(event.currentTarget.value)} />
      <button type="submit" className={inputClass} disabled={!draft.trim() || draft.trim() === name}>{studioName(action)}</button>
    </form>
  );
}

function InputSubset({ input, editor, onEdit }: { input: StudioRunInputView; editor: InputSubsetView; onEdit: (edit: StudioInputEdit) => void }) {
  return (
    <div className="space-y-2 rounded bg-shell-subtle p-2">
      {input.actions.includes("value") && editor.sweeps.map((sweep) => {
        const term = editor.values.find(({ sweepId }) => sweepId === sweep.id);
        return (
          <fieldset key={sweep.id} className="flex flex-wrap gap-2">
            <legend>{sweep.label}</legend>
            {sweep.values.map(({ id, label }) => (
              <label key={id} className="flex gap-1">
                <input type="checkbox" checked={term?.valueIds.includes(id) ?? false} onChange={(event) => onEdit({
                  kind: "value", sweepId: sweep.id, valueId: id, included: event.currentTarget.checked,
                })} />
                {label}
              </label>
            ))}
          </fieldset>
        );
      })}
      {input.actions.includes("row") && editor.rows.length > 0 && (
        <details>
          <summary>{studioName("rows")}</summary>
          <div className="max-h-36 overflow-auto">
            {editor.rows.map(({ id, label }) => (
              <label key={id} className="flex gap-1">
                <input type="checkbox" checked={editor.rowIds.includes(id)} onChange={(event) => onEdit({ kind: "row", rowId: id, included: event.currentTarget.checked })} />
                {label}
              </label>
            ))}
          </div>
        </details>
      )}
      {input.actions.includes("measure") && editor.metrics.length > 0 && <MeasureEditor editor={editor} onEdit={(rule) => onEdit({ kind: "measure", rule })} />}
      <details>
        <summary>{studioName("members")}</summary>
        <ul className="max-h-36 overflow-auto">
          {editor.members.map((member) => "rank" in member
            ? <li key={member.rank}>{studioName("rank")} {member.rank}</li>
            : <li key={member.id}>{member.label}</li>)}
        </ul>
      </details>
      {input.actions.includes("toRows") && (
        <button type="button" className={inputClass} onClick={() => onEdit({ kind: "toRows" })}>{studioName("toRows")}</button>
      )}
      {input.actions.includes(editor.name ? "rename" : "saveSubset") && (
        <SubsetNameForm key={editor.name} name={editor.name} onEdit={onEdit} />
      )}
    </div>
  );
}

export function StudioRunInputs({ inputs, onEdit }: { inputs: readonly StudioRunInputView[]; onEdit: (id: string, edit: StudioInputEdit) => void }) {
  return (
    <section aria-label={studioName("inputs")} className="space-y-2">
      <h2 className="font-semibold">{studioName("inputs")}</h2>
      {inputs.map((input) => (
        <div key={input.id} className="space-y-2 rounded border border-shell-divider p-2">
          <div className="flex flex-wrap items-baseline gap-2">
            <span className="font-medium">{input.label}</span>
            <select
              aria-label={studioAccessibleName("input", input.label)} value={input.choiceId ?? ""} className={inputClass}
              disabled={!input.actions.includes("choose")}
              onChange={(event) => onEdit(input.id, { kind: "choose", choiceId: event.currentTarget.value })}
            >
              <option value="" disabled>{studioName("choose")}</option>
              {([["step", "steps"], ["run", "runs"], ["subset", "subset"]] as const).map(([kind, name]) => {
                const choices = input.choices.filter((choice) => choice.kind === kind);
                return choices.length ? (
                  <optgroup key={kind} label={studioName(name)}>
                    {choices.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
                  </optgroup>
                ) : null;
              })}
            </select>
            {input.reason && <p className="text-shell-status-failure">{input.reason}</p>}
          </div>
          {input.subset && (
            <p className="text-shell-muted">{[...input.subset.terms, studioCountOf(input.subset.count, input.subset.total)].join(" · ")}</p>
          )}
          {input.editor && <InputSubset input={input} editor={input.editor} onEdit={(edit) => onEdit(input.id, edit)} />}
        </div>
      ))}
    </section>
  );
}
