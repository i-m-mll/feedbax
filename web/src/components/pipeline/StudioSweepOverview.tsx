import { studioAccessibleName, studioName } from "@/components/activity/studioVocabulary";
import { Button } from "@/components/ui/Button";
import { Select } from "@/components/ui/Select";
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
  readonly removable?: boolean;
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
          <label className="flex items-center gap-2">
            {studioName("combination")}
            <Select disabled={!combination.onChange} value={combination.id} onValueChange={(id) => combination.onChange?.(id)}
              items={combination.options.map(({ id, label }) => ({ value: id, label }))} />
          </label>
        )}
        {onToRows && <Button onClick={onToRows}>{studioName("toRows")}</Button>}
      </div>
      {axes.length > 0 && <ul className="divide-y divide-shell-divider rounded-surface border border-shell-divider">
        {axes.map((axis, index) => (
          <li className="p-2" key={axis.id}>
            <div className="flex flex-wrap items-center gap-2">
              <strong>{axis.label}</strong>
              {onMove && index > 0 && (
                <Button aria-label={studioAccessibleName("moveEarlier", axis.label)} onClick={() => onMove(axis.id, axes[index - 1].id)}>{studioName("moveEarlier")}</Button>
              )}
              {onMove && index + 1 < axes.length && (
                <Button aria-label={studioAccessibleName("moveLater", axis.label)} onClick={() => onMove(axis.id, axes[index + 2]?.id ?? null)}>{studioName("moveLater")}</Button>
              )}
            </div>
            <ul className="mt-1 grid gap-1 sm:grid-cols-2">
              {axis.values.map((value) => (
                <li key={value.id} className="rounded-row bg-shell-subtle p-1.5">
                  <span className="font-medium">{value.label}</span>
                  <Changes entries={value.changes} />
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ul>}
      {exceptions.length > 0 && (
        <div className="space-y-1">
          <h3 className="font-medium">{studioName("exceptions")}</h3>
          <ul className="divide-y divide-shell-divider rounded-surface border border-shell-divider">
            {exceptions.map((exception) => (
              <li key={exception.id} className="space-y-1 p-2">
                <div className="flex items-center justify-between gap-2">
                  <span>{[...exception.terms, exception.count].join(" · ")}</span>
                  {onRemoveException && exception.removable !== false &&
                    <Button onClick={() => onRemoveException(exception.id)}>{studioName("remove")}</Button>}
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
        <li key={id}><Button onClick={() => onChoose(id)}>{label}</Button></li>
      ))}
    </ul>
  );
}
