import { useId } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
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

/** The binding of an Analyze step's input role to a step's output or a document of the role's kind. */
export function AnalyzeRoleBinding({ role, kind, binding, project, steps, picking, onBind }: {
  role: string;
  kind: RoleKind;
  binding: InputBinding | null | undefined;
  project: AuthoredProject;
  steps: readonly PlanStepView[];
  picking: ElementSelectionStore;
  onBind?: (binding: InputBinding) => void;
}) {
  const labelId = useId();
  const label = humanizeLabel(role);
  return <FieldRow label={label} labelId={labelId}>
    <ReferenceField label={label} aria-labelledby={labelId} picking={picking}
      groups={roleGroups(project, steps, kind)} value={boundChoice(binding, steps)}
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
export function AnalyzeElementInputs({ document, element, roles, inputs, declarations, outputs, picking, onWire }: {
  document: string;
  element: string;
  roles: readonly { name: string; kind: RoleKind }[];
  inputs: Readonly<Record<string, Source>>;
  declarations: Readonly<Record<string, Declaration>>;
  /** Each other operation's dataset products. */
  outputs: readonly { element: string; products: readonly string[] }[];
  picking: ElementSelectionStore;
  onWire?: (role: string, source: Source) => void;
}) {
  const labelId = useId();
  const named = (id: string) => ({ kind: 'element', address: { document, element: { kind: 'named', id } } }) as const;
  return <>{roles.map(({ name, kind }) => {
    const documentInputs = Object.entries(declarations).filter(([, declaration]) => declaration.kind === kind)
      .map(([input]): ReferenceChoice => ({ id: sourceId({ input }), label: input, pick: named(input) }));
    const products = kind !== 'dataset' ? [] : outputs.filter((output) => output.element !== element)
      .flatMap(({ element: source, products }) => products.map((product, index): ReferenceChoice => ({
        id: sourceId({ element: source, product }), label: source + '.' + product, ...(index === 0 && { pick: named(source) }),
      })));
    const choices = new Map([...documentInputs, ...products].map((choice) => [choice.id, choice]));
    const wired = inputs[name];
    const sources = new Map<string, Source>([
      ...Object.keys(declarations).map((input): [string, Source] => [sourceId({ input }), { input }]),
      ...outputs.flatMap(({ element: source, products }) =>
        products.map((product): [string, Source] => [sourceId({ element: source, product }), { element: source, product }])),
    ]);
    const label = humanizeLabel(name);
    return <FieldRow key={name} label={label} labelId={`${labelId}-${name}`}>
      <ReferenceField label={label} aria-labelledby={`${labelId}-${name}`} picking={picking}
        groups={[
          { label: studioName('inputs'), choices: documentInputs },
          { label: studioName('operations'), choices: products },
        ]}
        value={wired ? choices.get(sourceId(wired)) ?? { id: sourceId(wired), label: 'input' in wired ? wired.input
          : wired.element + '.' + wired.product } : null}
        onChoose={onWire && ((choice) => {
          const source = sources.get(choice.id);
          if (source) onWire(name, source);
        })} />
    </FieldRow>;
  })}</>;
}
