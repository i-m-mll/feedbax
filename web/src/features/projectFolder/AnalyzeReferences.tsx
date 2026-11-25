import { Sigma } from 'lucide-react';
import { useId } from 'react';
import { studioName } from '@/components/activity/studioVocabulary';
import { FieldRow } from '@/components/ui/FieldRow';
import { ReferenceField, type ReferenceChoice } from '@/components/values/ReferenceField';
import { humanizeLabel } from '@/components/values/humanizeLabel';
import type { ElementSelectionStore } from '@/features/projectWindow/elementSelection';
import type { AnalysisDocument, OperationDataset, StepInput } from '@/generated/studioContracts';

type Declaration = NonNullable<AnalysisDocument['inputs']>[string];
type Source = StepInput | OperationDataset;
export type RoleKind = Declaration['kind'];

/** An operation's output in another Analyze card, shown as `card/operation.product`. */
export interface CardOutput {
  readonly step: string;
  readonly element: string;
  readonly product: string;
  readonly name: string;
}

export const outputChoiceId = ({ step, element, product }: Pick<CardOutput, 'step' | 'element' | 'product'>) =>
  JSON.stringify(['output', step, element, product]);

export function outputChoice(output: CardOutput): ReferenceChoice {
  return { id: outputChoiceId(output), label: output.name, glyph: <Sigma /> };
}

const sourceId = (source: Source) => JSON.stringify('input' in source ? [source.input] : [source.element, source.product]);

/**
 * An operation's or figure's input roles, each wired to one of the document's inputs or another operation's output
 * of the role's kind. Picking fills a role from an element on the canvas.
 */
export function AnalyzeElementInputs({ document, element, roles, inputs, declarations, outputs, sources: fitting, picking, refusal, onWire }: {
  document: string;
  element: string;
  roles: readonly { name: string; kind: RoleKind }[];
  /** The sources that fit each dataset role, as the service reviews a wire: a dataset role offers only these. */
  sources: Readonly<Record<string, readonly Source[]>>;
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
    const fits = new Set((fitting[name] ?? []).map(sourceId));
    const offered = (choice: ReferenceChoice) => kind !== 'dataset' || fits.has(choice.id);
    const documentInputs = Object.entries(declarations).filter(([, declaration]) => declaration.kind === kind)
      .map(([input]): ReferenceChoice => ({ id: sourceId({ input }), label: input, pick: named(input) })).filter(offered);
    const products = kind !== 'dataset' ? [] : outputs.filter((output) => output.element !== element)
      .flatMap(({ element: source, name: shown, products }) => products
        .map((product): ReferenceChoice => ({ id: sourceId({ element: source, product }), label: shown + '.' + product }))
        .filter(offered).map((choice, index) => index === 0 ? { ...choice, pick: named(source) } : choice));
    const choices = new Map([...documentInputs, ...products].map((choice) => [choice.id, choice]));
    const wired = inputs[name];
    const sources = new Map<string, Source>([
      ...Object.keys(declarations).map((input): [string, Source] => [sourceId({ input }), { input }]),
      ...outputs.flatMap(({ element: source, products }) =>
        products.map((product): [string, Source] => [sourceId({ element: source, product }), { element: source, product }])),
    ]);
    const label = humanizeLabel(name);
    const refused = refusal?.(name);
    return <FieldRow key={name} label={label} labelId={`${labelId}-${name}`}>
      <ReferenceField label={label} aria-labelledby={`${labelId}-${name}`} picking={picking} problem={refused}
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
