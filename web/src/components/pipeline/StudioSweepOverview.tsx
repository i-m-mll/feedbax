import { studioAccessibleName, studioName } from "@/components/activity/studioVocabulary";
import type { StudioDefinitionField } from "./StudioDefinitionForm";

interface SweepAxisView {
  readonly id: string;
  readonly label: string;
  readonly values: readonly {
    readonly id: string;
    readonly label: string;
    readonly changes: readonly { readonly id: string; readonly label: string; readonly value: string }[];
  }[];
}

interface ExceptionView {
  readonly id: string;
  readonly terms: readonly string[];
  readonly count: number;
  readonly changes: readonly { readonly id: string; readonly label: string; readonly value: string }[];
}

interface SweepOverviewProps {
  axes: readonly SweepAxisView[];
  exceptions: readonly ExceptionView[];
  combination?: { id: string; options: readonly { id: string; label: string }[]; onChange?: (id: string) => void };
  onMove?: (id: string, before: string | null) => void;
  onRemoveException?: (id: string) => void;
  onToRows?: () => void;
}

const commandClass = "rounded border border-shell-control px-2 py-1 text-sm";

function Changes({ entries }: { entries: ExceptionView["changes"] }) {
  return (
    <dl className="flex flex-wrap gap-x-2 gap-y-1">
      {entries.map(({ id, label, value }) => (
        <div key={id} className="flex gap-1"><dt>{label}</dt><dd>{value}</dd></div>
      ))}
    </dl>
  );
}

export function StudioSweepOverview({ axes, exceptions, combination, onMove, onRemoveException, onToRows }: SweepOverviewProps) {
  return (
    <section aria-label={studioName("sweep")} className="space-y-2">
      <div className="flex flex-wrap items-baseline gap-2">
        <h2 className="font-semibold">{studioName("sweep")}</h2>
        {combination && (
          <label className="flex items-baseline gap-2">
            {studioName("combination")}
            <select className={commandClass} disabled={!combination.onChange} value={combination.id} onChange={(event) => combination.onChange?.(event.currentTarget.value)}>
              {combination.options.map(({ id, label }) => <option key={id} value={id}>{label}</option>)}
            </select>
          </label>
        )}
        {onToRows && <button type="button" className={commandClass} onClick={onToRows}>{studioName("toRows")}</button>}
      </div>
      <ul className="divide-y divide-shell-divider rounded border border-shell-divider">
        {axes.map((axis, index) => (
          <li className="p-2" key={axis.id}>
            <div className="flex flex-wrap items-baseline gap-2">
              <strong>{axis.label}</strong>
              {onMove && index > 0 && (
                <button type="button" className={commandClass} aria-label={studioAccessibleName("moveEarlier", axis.label)} onClick={() => onMove(axis.id, axes[index - 1].id)}>{studioName("moveEarlier")}</button>
              )}
              {onMove && index + 1 < axes.length && (
                <button type="button" className={commandClass} aria-label={studioAccessibleName("moveLater", axis.label)} onClick={() => onMove(axis.id, axes[index + 2]?.id ?? null)}>{studioName("moveLater")}</button>
              )}
            </div>
            <ul className="mt-1 grid gap-1 sm:grid-cols-2">
              {axis.values.map((value) => (
                <li key={value.id} className="rounded bg-shell-subtle p-1.5">
                  <span className="font-medium">{value.label}</span>
                  <Changes entries={value.changes} />
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
      {exceptions.length > 0 && (
        <div className="space-y-1">
          <h3 className="font-medium">{studioName("exceptions")}</h3>
          <ul className="divide-y divide-shell-divider rounded border border-shell-divider">
            {exceptions.map((exception) => (
              <li key={exception.id} className="space-y-1 p-2">
                <div className="flex items-baseline justify-between gap-2">
                  <span>{[...exception.terms, exception.count].join(" · ")}</span>
                  {onRemoveException && <button type="button" className={commandClass} onClick={() => onRemoveException(exception.id)}>{studioName("remove")}</button>}
                </div>
                <Changes entries={exception.changes} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

export function StudioSweepPicker({ fields, onChoose }: { fields: readonly StudioDefinitionField[]; onChoose: (id: string) => void }) {
  return (
    <ul aria-label={studioName("newSweep")} className="space-y-1">
      {fields.filter(({ actions, resolved, sweep }) => actions.includes("toSweep") && resolved.length === 1 && !sweep).map(({ id, label }) => (
        <li key={id}><button type="button" className={commandClass} onClick={() => onChoose(id)}>{label}</button></li>
      ))}
    </ul>
  );
}
