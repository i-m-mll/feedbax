import { useId } from 'react';
import { studioCountOf, studioName } from '@/components/activity/studioVocabulary';
import { FieldRow } from '@/components/ui/FieldRow';
import { ReferenceField, type ReferenceChoice, type ReferenceGroup } from '@/components/values/ReferenceField';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { PlanStepView } from '@/features/pipeline/planView';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import {
  StepReferenceSchema, type AnalysisDocument, type InputBinding, type OperationDataset, type StepInput,
} from '@/generated/studioContracts';
import type { AuthoredProject } from './authoredEdits';
import {
  documentChoice, documentChoiceId, documentGroup, stepChoice, stepChoiceId, stepGroup,
} from './referenceChoices';
import { InputSelections } from './InputSelections';
import { changedSelections, stepSelections } from './selectionEdits';

type Declaration = NonNullable<AnalysisDocument['inputs']>[string];
type Source = StepInput | OperationDataset;
export type RoleKind = Declaration['kind'];

/** What a step's input role can be bound to, for the role's declared kind. */
function roleGroups(project: AuthoredProject, steps: readonly PlanStepView[], kind: RoleKind): ReferenceGroup[] {
  switch (kind) {
    case 'dataset': return [stepGroup(steps.filter((step) => step.kind === 'evaluate'))];
    case 'model': return [documentGroup(project, 'model'), stepGroup(steps.filter((step) => step.kind === 'train'))];
    case 'task': return [documentGroup(project, 'task')];
  }
}

function boundChoice(binding: InputBinding | null | undefined, steps: readonly PlanStepView[]): ReferenceChoice | null {
  const documentBinding = binding as { kind?: unknown; source?: { document?: unknown } } | null | undefined;
  const document = documentBinding?.source?.document;
  if (typeof document === 'string') return documentChoice(documentBinding?.kind === 'task' ? 'task' : 'model', document);
  const reference = StepReferenceSchema.safeParse(binding);
  if (!reference.success) return null;
  const step = steps.find(({ id }) => id === reference.data.step);
  return step ? stepChoice(step) : { id: stepChoiceId(reference.data.step), label: reference.data.step };
}

/** The binding of an Analyze step's input role to a step's output or a document of the role's kind. A dataset role takes a list. */
export function AnalyzeRoleBinding({ role, kind, binding, project, steps, inputs, picking, refusal, problem, onBind }: {
  role: string;
  /** Why the last binding was refused. */
  refusal?: string;
  kind: RoleKind;
  binding: InputBinding | null | undefined;
  project: AuthoredProject;
  steps: readonly PlanStepView[];
  /** The plan's inputs of the step, whose subsets the selections' rows show. */
  inputs?: PlanStepView['inputs'];
  picking: ElementSelectionStore;
  /** Why the plan refuses selection `index` of the role. */
  problem?: (index: number) => string | undefined;
  onBind?: (binding: InputBinding) => void;
}) {
  const labelId = useId();
  const label = humanizeLabel(role);
  const groups = roleGroups(project, steps, kind);
  const selections = kind === 'dataset' ? stepSelections(binding) : undefined;
  if (selections) {
    const listed = Array.isArray(binding);
    const stepOf = (choice: { id: string }) => steps.find(({ id }) => stepChoiceId(id) === choice.id);
    return <InputSelections label={label} labelId={labelId} groups={groups} picking={picking} refusal={refusal}
      rows={selections.map((reference, index) => {
        const subset = inputs?.find(({ id }) => id === (listed ? `${role}.${index}` : role))?.subset;
        return {
          value: boundChoice(reference, steps),
          problem: listed ? problem?.(index) : undefined,
          detail: subset && <div className="px-control-inline text-shell-muted">
            {[...subset.terms, studioCountOf(subset.count, subset.total)].join(' · ')}
          </div>,
        };
      })}
      onChange={onBind && ((gesture) => {
        if (gesture.kind === 'remove') return onBind(changedSelections(selections, gesture));
        const step = stepOf(gesture.choice);
        if (!step) return;
        if (gesture.kind === 'add') onBind(changedSelections(selections, { kind: 'add', reference: { step: step.id } }));
        else if (step.id !== selections[gesture.index].step) {
          onBind(changedSelections(selections, { kind: 'replace', index: gesture.index, step: step.id }));
        }
      })} />;
  }
  return <FieldRow label={label} labelId={labelId} below={refusal && <p role="alert">{refusal}</p>}>
    <ReferenceField label={label} aria-labelledby={labelId} picking={picking}
      groups={groups} value={boundChoice(binding, steps)}
      onChoose={onBind && ((choice) => {
        const step = steps.find(({ id }) => stepChoiceId(id) === choice.id);
        if (step) onBind(kind === 'model' ? { step: step.id, checkpoint: 'last' } : { step: step.id });
        else if (choice.id === documentChoiceId(choice.label)) {
          onBind(kind === 'task'
            ? { kind: 'task', source: { document: choice.label } }
            : { kind: 'initial_model', source: { document: choice.label } });
        }
      })} />
  </FieldRow>;
}

const sourceId = (source: Source) => JSON.stringify('input' in source ? [source.input] : [source.element, source.product]);

/**
 * An operation's or figure's input roles, each wired to one of the document's inputs or another operation's output
 * of the role's kind. Picking fills a role from an element on the canvas.
 */
export function AnalyzeElementInputs({ document, element, roles, inputs, declarations, outputs, picking, refusal, onWire }: {
  document: string;
  element: string;
  roles: readonly { name: string; kind: RoleKind }[];
  inputs: Readonly<Record<string, Source>>;
  declarations: Readonly<Record<string, Declaration>>;
  /** Each other operation's shown name and dataset products. */
  outputs: readonly { element: string; name: string; products: readonly string[] }[];
  picking: ElementSelectionStore;
  /** Why a role's last wiring was refused. */
  refusal?: (role: string) => string | undefined;
  onWire?: (role: string, source: Source) => void;
}) {
  const labelId = useId();
  const named = (id: string) => ({ kind: 'element', address: { document, element: { kind: 'named', id } } }) as const;
  return <>{roles.map(({ name, kind }) => {
    const documentInputs = Object.entries(declarations).filter(([, declaration]) => declaration.kind === kind)
      .map(([input]): ReferenceChoice => ({ id: sourceId({ input }), label: input, pick: named(input) }));
    const products = kind !== 'dataset' ? [] : outputs.filter((output) => output.element !== element)
      .flatMap(({ element: source, name: shown, products }) => products.map((product, index): ReferenceChoice => ({
        id: sourceId({ element: source, product }), label: shown + '.' + product, ...(index === 0 && { pick: named(source) }),
      })));
    const choices = new Map([...documentInputs, ...products].map((choice) => [choice.id, choice]));
    const wired = inputs[name];
    const sources = new Map<string, Source>([
      ...Object.keys(declarations).map((input): [string, Source] => [sourceId({ input }), { input }]),
      ...outputs.flatMap(({ element: source, products }) =>
        products.map((product): [string, Source] => [sourceId({ element: source, product }), { element: source, product }])),
    ]);
    const label = humanizeLabel(name);
    const refused = refusal?.(name);
    return <FieldRow key={name} label={label} labelId={`${labelId}-${name}`} below={refused && <p role="alert">{refused}</p>}>
      <ReferenceField label={label} aria-labelledby={`${labelId}-${name}`} picking={picking}
        groups={[
          { label: studioName('inputs'), choices: documentInputs },
          { label: studioName('operations'), choices: products },
        ]}
        value={wired ? choices.get(sourceId(wired)) ?? { id: sourceId(wired), label: 'input' in wired ? wired.input
          : (outputs.find((output) => output.element === wired.element)?.name ?? wired.element) + '.' + wired.product } : null}
        onChoose={onWire && ((choice) => {
          const source = sources.get(choice.id);
          if (source) onWire(name, source);
        })} />
    </FieldRow>;
  })}</>;
}
