import { useId } from "react";
import { Circle, CircleDot, Plus } from "lucide-react";
import { studioAccessibleName, studioName } from "@/components/activity/studioVocabulary";
import type { JsonPrimitive } from "@/generated/studioContracts";
import type { PipelineSelection } from "@/features/pipeline/selection";

export type StudioFieldControl =
  | { readonly kind: "number"; readonly min?: number; readonly max?: number; readonly step?: number }
  | { readonly kind: "boolean" }
  | { readonly kind: "choice"; readonly options: readonly { value: JsonPrimitive; label: string }[] };

interface FieldValue {
  readonly id: string;
  readonly value: JsonPrimitive;
  readonly label?: string;
  readonly source?: string;
  readonly changedFrom?: { readonly label: string };
  readonly selected?: boolean;
  readonly reason?: string;
}

export type FieldSweep =
  | { readonly kind: "list"; readonly id: string; readonly label: string; readonly values: readonly FieldValue[] }
  | { readonly kind: "range"; readonly id: string; readonly label: string; readonly first: number; readonly last: number; readonly count: number; readonly scale: "linear" | "logarithmic" }
  | { readonly kind: "sampler"; readonly id: string; readonly label: string; readonly distribution: string; readonly count: number; readonly seed: number };

export interface StudioDefinitionField {
  readonly id: string;
  readonly label: string;
  readonly unit?: string;
  readonly control?: StudioFieldControl;
  readonly resolved: readonly FieldValue[];
  readonly sweep?: FieldSweep;
  readonly actions: readonly FieldEdit["kind"][];
}

export type FieldEdit =
  | { readonly kind: "set"; readonly value: JsonPrimitive }
  | { readonly kind: "toSweep"; readonly value: JsonPrimitive }
  | { readonly kind: "setSweepValue"; readonly sweepId: string; readonly valueId: string; readonly value: JsonPrimitive }
  | { readonly kind: "addValue"; readonly sweepId: string }
  | { readonly kind: "toField"; readonly sweepId: string; readonly keepValueId: string }
  | { readonly kind: "toList"; readonly sweepId: string };

export interface StudioFieldCommand {
  readonly fieldId: string;
  readonly selection: PipelineSelection;
  readonly edit: FieldEdit;
}

interface FormProps {
  fields: readonly StudioDefinitionField[];
  selection: PipelineSelection;
  onEdit: (command: StudioFieldCommand) => void;
  onChooseSweepValue?: (sweepId: string, valueId: string) => void;
}

const controlClass = "min-w-0 rounded border border-slate-300 bg-white px-2 py-1 disabled:opacity-60";

function scalarText(value: JsonPrimitive) {
  if (typeof value === "boolean") return studioName(value ? "true" : "false");
  return value === null ? "—" : String(value);
}

function ScalarEditor({ control, name, value, onChange }: {
  control?: StudioFieldControl;
  name: string;
  value: FieldValue;
  onChange?: (value: JsonPrimitive) => void;
}) {
  if (!control || !onChange) return <span className="tabular-nums">{value.label ?? scalarText(value.value)}</span>;
  if (control.kind === "boolean") {
    return <input type="checkbox" aria-label={studioAccessibleName("edit", name)} checked={value.value === true} onChange={(event) => onChange(event.currentTarget.checked)} />;
  }
  if (control.kind === "choice") {
    const selected = control.options.findIndex((option) => option.value === value.value);
    return (
      <select
        className={controlClass} aria-label={studioAccessibleName("edit", name)} value={selected < 0 ? "" : selected}
        onChange={(event) => {
          const option = control.options[Number(event.currentTarget.value)];
          if (option) onChange(option.value);
        }}
      >
        {selected < 0 && <option value="" disabled>{value.label ?? scalarText(value.value)}</option>}
        {control.options.map((option, index) => <option key={index} value={index}>{option.label}</option>)}
      </select>
    );
  }
  return (
    <input
      key={String(value.value)}
      type="number" aria-label={studioAccessibleName("edit", name)} className={`${controlClass} w-28 tabular-nums`}
      defaultValue={typeof value.value === "number" ? value.value : ""}
      min={control.min} max={control.max} step={control.step ?? "any"}
      onBlur={({ currentTarget }) => {
        const next = currentTarget.valueAsNumber;
        if (currentTarget.validity.valid && Number.isFinite(next) && next !== value.value) onChange(next);
      }}
      onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); } }}
    />
  );
}

function ValueSource({ value }: { value: FieldValue }) {
  return (
    <>
      {value.changedFrom && <span className="text-slate-600"><s>{value.changedFrom.label}</s> →</span>}
      {value.source && <span className="rounded bg-slate-100 px-1.5 text-xs text-slate-700">{value.source}</span>}
      {value.reason && <span className="text-xs text-red-700">{value.reason}</span>}
    </>
  );
}

function RuleField({ sweep, canList, onEdit }: { sweep: Exclude<FieldSweep, { kind: "list" }>; canList: boolean; onEdit: (edit: FieldEdit) => void }) {
  const entries = sweep.kind === "range"
    ? ([["first", sweep.first], ["last", sweep.last], ["count", sweep.count], ["scale", studioName(sweep.scale)]] as const)
    : ([["distribution", sweep.distribution], ["count", sweep.count], ["seed", sweep.seed]] as const);
  return (
    <div className="flex flex-wrap items-baseline gap-2">
      <span className="font-medium">{sweep.label}</span>
      <dl className="flex flex-wrap gap-2">
        {entries.map(([name, value]) => (
          <div key={name} className="flex gap-1 tabular-nums"><dt>{studioName(name)}</dt><dd>{value}</dd></div>
        ))}
      </dl>
      {canList && (
        <button type="button" className={controlClass} onClick={() => onEdit({ kind: "toList", sweepId: sweep.id })}>
          {studioName("toList")}
        </button>
      )}
    </div>
  );
}

export function StudioDefinitionFieldEditor({ field, selection, onEdit, onChooseSweepValue }: Omit<FormProps, "fields"> & { field: StudioDefinitionField }) {
  const labelId = useId();
  const can = (kind: FieldEdit["kind"]) => field.actions.includes(kind);
  const edit = (command: FieldEdit) => onEdit({ fieldId: field.id, selection, edit: command });
  const sweep = field.sweep;
  const onlyValue = field.resolved.length === 1 ? field.resolved[0] : undefined;
  return (
    <div className="grid gap-1 rounded border border-slate-200 p-2 sm:grid-cols-[minmax(8rem,1fr)_3fr]" role="group" aria-labelledby={labelId}>
      <div className="flex items-start gap-1">
        <span id={labelId} className="font-medium">{field.label}</span>
        {!sweep && onlyValue && can("toSweep") && (
          <button type="button" className="rounded p-0.5" aria-label={studioAccessibleName("toSweep", field.label)} onClick={() => edit({ kind: "toSweep", value: onlyValue.value })}>
            <Circle aria-hidden="true" size={15} />
          </button>
        )}
      </div>
      {!sweep && (
        <ul className="space-y-1">
          {field.resolved.map((value) => (
            <li key={value.id} className="flex flex-wrap items-center gap-2">
              <ValueSource value={value} />
              <ScalarEditor name={field.label} value={value} control={field.control} onChange={onlyValue && can("set") ? (next) => edit({ kind: "set", value: next }) : undefined} />
              {field.unit && <span className="text-slate-700">{field.unit}</span>}
            </li>
          ))}
        </ul>
      )}
      {sweep?.kind === "list" && (
        <ul className="flex flex-wrap gap-2">
          {sweep.values.map((value) => {
            const label = value.label ?? scalarText(value.value);
            const name = `${field.label} ${label}`;
            return (
              <li key={value.id} data-selected={value.selected || undefined} className="flex flex-wrap items-center gap-1 rounded p-1 data-[selected=true]:bg-violet-100">
                {onChooseSweepValue && (
                  <button type="button" aria-pressed={value.selected ?? false} aria-label={studioAccessibleName("select", name)} onClick={() => onChooseSweepValue(sweep.id, value.id)} className="rounded p-0.5">
                    <CircleDot aria-hidden="true" size={15} />
                  </button>
                )}
                <ValueSource value={value} />
                <ScalarEditor name={name} value={value} control={field.control} onChange={can("setSweepValue") ? (next) => edit({ kind: "setSweepValue", sweepId: sweep.id, valueId: value.id, value: next }) : undefined} />
                {can("toField") && (
                  <button type="button" className={controlClass} aria-label={studioAccessibleName("toField", name)} onClick={() => edit({ kind: "toField", sweepId: sweep.id, keepValueId: value.id })}>
                    {studioName("toField")}
                  </button>
                )}
              </li>
            );
          })}
          {can("addValue") && (
            <li>
              <button type="button" className={controlClass} aria-label={studioAccessibleName("addValue", field.label)} onClick={() => edit({ kind: "addValue", sweepId: sweep.id })}>
                <Plus aria-hidden="true" size={15} />
              </button>
            </li>
          )}
        </ul>
      )}
      {sweep && sweep.kind !== "list" && <RuleField sweep={sweep} canList={can("toList")} onEdit={edit} />}
    </div>
  );
}

export function StudioDefinitionForm({ fields, ...props }: FormProps) {
  return (
    <section aria-label={studioName("definition")} className="space-y-2">
      <h2 className="font-semibold">{studioName("definition")}</h2>
      {fields.map((field) => <StudioDefinitionFieldEditor key={field.id} field={field} {...props} />)}
    </section>
  );
}
