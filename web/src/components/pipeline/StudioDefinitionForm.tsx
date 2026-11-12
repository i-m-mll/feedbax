import { useId, useState } from "react";
import { Circle, Info, Pencil, Plus } from "lucide-react";
import { studioAccessibleName, studioName } from "@/components/activity/studioVocabulary";
import type { JsonPrimitive } from "@/generated/studioContracts";
import { selectionButton, type PipelineSelection } from "@/features/pipeline/selection";

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
  | { readonly kind: "nameSweepValue"; readonly sweepId: string; readonly valueId: string; readonly label: string | null }
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
  onChooseSweepValue?: (sweepId: string, valueId: string, extend: boolean) => void;
}

const controlClass = "min-w-0 max-w-full rounded border border-shell-control bg-shell-surface px-1.5 py-0.5 disabled:opacity-60";

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

export function StudioDocumentPicker({ field, selection, onEdit }: Pick<FormProps, 'selection' | 'onEdit'> & {
  field: StudioDefinitionField;
}) {
  const value = field.resolved[0];
  if (!value) return null;
  const change = field.actions.includes('set') && field.resolved.length === 1
    ? (next: JsonPrimitive) => onEdit({ fieldId: field.id, selection, edit: { kind: 'set', value: next } })
    : undefined;
  return <ScalarEditor name={field.label} value={value} control={field.control} onChange={change} />;
}

function ValueSource({ value }: { value: FieldValue }) {
  return (
    <>
      {value.changedFrom && <span className="text-shell-muted"><s>{value.changedFrom.label}</s> →</span>}
      {value.source && <span className="rounded bg-shell-subtle px-1.5 text-xs text-shell-muted">{value.source}</span>}
      {value.reason && <span className="text-xs text-shell-status-failure">{value.reason}</span>}
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
  const valueEditorId = useId();
  const [editingValue, setEditingValue] = useState<string>();
  const edited = field.sweep?.kind === "list" ? field.sweep.values.find(({ id }) => id === editingValue) : undefined;
  const can = (kind: FieldEdit["kind"]) => field.actions.includes(kind);
  const edit = (command: FieldEdit) => onEdit({ fieldId: field.id, selection, edit: command });
  const sweep = field.sweep;
  const onlyValue = field.resolved.length === 1 ? field.resolved[0] : undefined;
  return (
    <div className="studio-definition-field grid gap-1 border-t border-shell-divider py-1.5" role="group" aria-labelledby={labelId}>
      <div className="flex items-start gap-1">
        <span id={labelId} className="min-w-0 font-medium">{field.label}</span>
        {!sweep && onlyValue && can("toSweep") && (
          <button type="button" className="rounded p-0.5" aria-label={studioAccessibleName("toSweep", field.label)} onClick={() => edit({ kind: "toSweep", value: onlyValue.value })}>
            <Circle aria-hidden="true" size={15} />
          </button>
        )}
      </div>
      {!sweep && (
        <ul className="space-y-1">
          {field.resolved.map((value) => (
            <li key={value.id} className="flex min-w-0 flex-wrap items-center gap-1">
              <ScalarEditor name={field.label} value={value} control={field.control} onChange={onlyValue && can("set") ? (next) => edit({ kind: "set", value: next }) : undefined} />
              {field.unit && <span className="text-shell-muted">{field.unit}</span>}
              <ValueSource value={value} />
            </li>
          ))}
        </ul>
      )}
      {sweep?.kind === "list" && (
        <div className="min-w-0 space-y-1.5">
          <ul className="flex flex-wrap gap-1">
            {sweep.values.map((value) => {
              const label = value.label ?? scalarText(value.value);
              const name = `${field.label} ${label}`;
              const editable = can("nameSweepValue") || can("setSweepValue") || can("toField");
              const description = [value.source, value.changedFrom?.label, value.reason].filter(Boolean).join(" · ");
              const Icon = editable ? Pencil : Info;
              return (
                <li key={value.id} data-selected={value.selected || undefined}
                  className="flex max-w-full items-center rounded border border-shell-divider data-[selected=true]:border-shell-foreground data-[selected=true]:bg-shell-selected">
                  {onChooseSweepValue
                    ? <button type="button" aria-pressed={value.selected ?? false} aria-label={studioAccessibleName("select", name)}
                      aria-description={description || undefined} title={description || undefined}
                      {...selectionButton((extend) => onChooseSweepValue(sweep.id, value.id, extend))}
                      className="min-w-0 truncate px-1.5 py-0.5 tabular-nums">{label}</button>
                    : <span className="min-w-0 truncate px-1.5 py-0.5 tabular-nums">{label}</span>}
                  {(editable || description) && <button type="button" aria-label={studioAccessibleName(editable ? "edit" : "details", name)}
                    aria-expanded={edited?.id === value.id} aria-controls={valueEditorId}
                    onClick={() => setEditingValue(edited?.id === value.id ? undefined : value.id)}
                    className="shrink-0 rounded p-1 hover:bg-shell-hover">
                    <Icon aria-hidden="true" size={12} />
                  </button>}
                </li>
              );
            })}
            {can("addValue") && <li>
              <button type="button" className={controlClass} aria-label={studioAccessibleName("addValue", field.label)}
                onClick={() => edit({ kind: "addValue", sweepId: sweep.id })}>
                <Plus aria-hidden="true" size={15} />
              </button>
            </li>}
          </ul>
          {edited && <div id={valueEditorId} className="flex min-w-0 flex-wrap items-center gap-1" key={edited.id}>
            {can("nameSweepValue") && (
              <input key={edited.label ?? ""} type="text" className={`${controlClass} w-24`}
                aria-label={studioAccessibleName("name", `${field.label} ${edited.label ?? scalarText(edited.value)}`)}
                placeholder={scalarText(edited.value)} defaultValue={edited.label ?? ""}
                onBlur={({ currentTarget }) => {
                  const next = currentTarget.value.trim() || null;
                  if (next !== (edited.label ?? null)) edit({ kind: "nameSweepValue", sweepId: sweep.id, valueId: edited.id, label: next });
                }}
                onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); event.currentTarget.blur(); } }}
              />
            )}
            <ScalarEditor name={`${field.label} ${edited.label ?? scalarText(edited.value)}`} value={{ ...edited, label: undefined }}
              control={field.control} onChange={can("setSweepValue") ? (next) => edit({ kind: "setSweepValue", sweepId: sweep.id, valueId: edited.id, value: next }) : undefined} />
            <ValueSource value={edited} />
            {can("toField") && (
              <button type="button" className={controlClass}
                aria-label={studioAccessibleName("toField", `${field.label} ${edited.label ?? scalarText(edited.value)}`)}
                onClick={() => edit({ kind: "toField", sweepId: sweep.id, keepValueId: edited.id })}>
                {studioName("toField")}
              </button>
            )}
          </div>}
        </div>
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
