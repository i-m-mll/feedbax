import { useState, type FormEvent } from 'react';
import { documentPayload, stepDocument } from './documentPayload';
import { studioName, studioRunStatus } from '@/components/activity/studioVocabulary';
import { ChangeRow, ProblemRow } from '@/features/projectWindow/StudioProblems';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Select } from '@/components/ui/Select';
import { TextArea, TextField } from '@/components/ui/TextField';
import { bindStudioCommand, commandAttributes, type StudioCommand, type StudioCommandId } from '@/features/projectWindow/studioCommands';
import type { PlanView } from '@/features/pipeline/planView';
import {
  ProjectValueEditSchema, ReportDocumentSchema, StepReferenceSchema, type ReportDocument, type StudioProjectFilePatch,
  type StepReference, type StudioSavedTrainKey, type StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { isReportFigureSource, reportFigureRole, type ReportFigureSource } from './createManagedStep';
import type { ProjectEditController, ProjectEditSnapshot } from './projectEditController';
import { savedRunId } from './projectWindowPlan';
import { SavedReportResult } from './SavedLightProducts';
import type { GestureFailures } from './refusals';

type Block = ReportDocument['blocks'][number];
type BlockAction = Extract<StudioCommandId, 'moveEarlier' | 'moveLater' | 'remove'>;

function figureLabel(source: ReportFigureSource) {
  const { step, row } = source.key;
  return step + ' · ' + source.name + (row === 'default' ? '' : ' · ' + row);
}

function figureAddress({ key, name, figure }: ReportFigureSource) {
  return JSON.stringify([key.variant, key.step, key.row, name, figure.product]);
}

/** A figure pin's IDs, independent of the displayed value labels. */
function figurePin(where: StepReference['where']) {
  if (where && ('subset' in where || where.rule || where.bound_ranks)) return undefined;
  return (where?.clauses ?? []).map((clause) => Object.entries(clause).sort(([left], [right]) => left.localeCompare(right))
    .map(([axis, values]) => [axis, values.map(({ id }) => id).sort()]));
}

/** Where a refusal lands: a field, a block by index, or the open dialog. */
type Place = 'title' | 'completeness' | `block:${number}` | 'dialog';

/** The report document a Report step holds, as the draft has it. */
export function stepReport({ draft, saved }: Pick<ProjectEditSnapshot, 'draft' | 'saved'>, stepId: string | undefined) {
  const step = stepId ? draft.steps?.[stepId] : undefined;
  const base = step?.kind === 'report' ? step.base : undefined;
  const document = stepDocument(base);
  const authored = document ? draft.documents?.[document] : undefined;
  const parsed = authored?.schema === 'feedbax.report_document.v1'
    ? ReportDocumentSchema.safeParse(documentPayload(draft.documents, saved.composed, document!)) : undefined;
  return { step, document, report: parsed?.success ? parsed.data : undefined };
}

export function useManagedReport({
  editor, snapshot, plan, projection, variant, stepId, run, wholeStep, focused, failures,
}: {
  editor: ProjectEditController;
  /** Where the step's refused gestures land. */
  failures: GestureFailures;
  snapshot: ProjectEditSnapshot;
  plan: PlanView;
  projection?: StudioSavedTrainRuns;
  variant: string | null;
  stepId?: string;
  run?: StudioSavedTrainKey;
  wholeStep: boolean;
  focused: boolean;
}) {
  const { draft, outside, pending, saved } = snapshot;
  const { step, document, report } = stepReport(snapshot, stepId);
  const planned = run && projection?.window_map?.rows.find(({ key }) => savedRunId(key) === savedRunId(run));
  const retained = planned?.light?.result ?? planned?.light?.previous;
  const previewUrl = retained?.products.report;
  const scope = JSON.stringify([saved.project_id, variant, stepId, document]);
  const [dialog, setDialog] = useState<{ scope: string; revision: string; kind: 'text' | 'figure' }>();
  const [figureChoice, setFigureChoice] = useState<string | null>(null);
  const [selection, setSelection] = useState<{ scope: string; index: number }>();
  /** A refused gesture's reason, on the field, block or open dialog it was made in. */
  const [refusal, setRefusal] = useState<{ scope: string; at: Place; reason: string }>();
  const shared = document && (plan.steps.some(({ id, documentUses }) => id !== stepId && documentUses?.includes(document))
    || Object.entries(draft.steps ?? {}).some(([id, { base }]) => id !== stepId && 'document' in base && base.document === document));
  const direct = !step?.sweep && !step?.deltas?.length;
  /** Whether its commands apply; a gesture made while the workspace is unsettled waits for it. */
  const applies = !!report && !!document && wholeStep && direct && !outside && !shared;
  const editable = applies && !pending;
  const figures = projection?.window_map?.analysis_elements?.filter(isReportFigureSource)
    .filter(({ key }) => key.variant === variant) ?? [];
  const inputs = step?.inputs ?? {};
  const opened = dialog?.scope === scope && dialog.revision === saved.revision && focused ? dialog.kind : undefined;
  const selected = selection?.scope === scope ? selection.index : undefined;

  function save(name: string, at: Place, patches: StudioProjectFilePatch['patches'], nextInputs?: typeof inputs) {
    if (!editable || !stepId || !document) return false;
    const land = (reason: string) => setRefusal({ scope, at, reason });
    failures.clear();
    setRefusal(undefined);
    try {
      const files: StudioProjectFilePatch[] = [{ address: { kind: 'document', id: document }, patches }];
      if (nextInputs) files.push({ address: { kind: 'step', id: stepId }, patches: [
        ProjectValueEditSchema.parse({ op: step?.inputs ? 'replace' : 'add', path: '/inputs', value: nextInputs }),
      ] });
      void editor.editFiles(name, files, { reviewedRevision: saved.revision }).then((settled) => {
        // The dialog has closed by the time the save settles, so its refusal goes to the step.
        if (settled.status === 'refused') failures.fail(settled.error, at === 'dialog' ? undefined : land);
      });
      return true;
    } catch (cause) {
      failures.fail(cause, land);
      return false;
    }
  }

  function setBlocks(name: string, at: Place, blocks: Block[], nextInputs?: typeof inputs) {
    return save(name, at, [{ op: 'replace', path: '/payload/blocks', value: blocks }], nextInputs);
  }
  const refusalAt = (at: Place) => refusal?.scope === scope && refusal.at === at
    ? <p role="alert">{refusal.reason}</p> : null;

  function changeBlock(index: number, block: Block) {
    const previous = report?.blocks[index];
    if (!previous) return;
    save('markdown' in block ? studioName('text') : studioName('caption'), `block:${index}`, [{
      op: 'replace', path: '/payload/blocks/' + index, value: block,
    }]);
  }

  function blockCommand(id: BlockAction, index: number) {
    const blocks = report?.blocks ?? [];
    const block = blocks[index];
    const target = index + (id === 'moveEarlier' ? -1 : 1);
    return bindStudioCommand(id, () => {
      if (id !== 'remove') {
        const reordered = blocks.slice();
        [reordered[index], reordered[target]] = [reordered[target], reordered[index]];
        if (setBlocks(studioName(id), `block:${index}`, reordered)) setSelection({ scope, index: target });
        return;
      }
      const remaining = blocks.filter((_, position) => position !== index);
      let nextInputs: typeof inputs | undefined;
      if ('figure' in block && Object.prototype.hasOwnProperty.call(inputs, block.figure) &&
          !remaining.some((item) => 'figure' in item && item.figure === block.figure)) {
        nextInputs = { ...inputs };
        delete nextInputs[block.figure];
      }
      if (setBlocks(studioName(id), `block:${index}`, remaining, nextInputs)) {
        setSelection(remaining.length ? { scope, index: Math.min(index, remaining.length - 1) } : undefined);
      }
    }, {
      enabled: applies && !!block && (id === 'remove' || target >= 0 && target < blocks.length),
      subject: block ? String(index + 1) : undefined,
    });
  }

  const blockActions = ['moveEarlier', 'moveLater', 'remove'] as const;
  const addText = bindStudioCommand('addText', () => {
    setRefusal(undefined);
    setDialog({ scope, revision: saved.revision, kind: 'text' });
  }, { enabled: applies });
  const addFigure = bindStudioCommand('addFigure', () => {
    setFigureChoice(null);
    setRefusal(undefined);
    setDialog({ scope, revision: saved.revision, kind: 'figure' });
  }, {
    enabled: applies && figures.length > 0,
  });
  const commands = focused && report ? [
    addText, addFigure,
    ...(selected === undefined ? [] : blockActions.map((id) => blockCommand(id, selected))),
  ] : [];

  function addBlock(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!opened || !report) return;
    const value = opened === 'text' ? new FormData(event.currentTarget).get('block') : figureChoice;
    if (typeof value !== 'string' || !value.trim()) return;
    let accepted: boolean;
    if (opened === 'text') {
      accepted = setBlocks(addText.label, 'dialog', [...report.blocks, { markdown: value }]);
    } else {
      const source = figures.find((figure) => figureAddress(figure) === value);
      if (!source) return;
      const existing = Object.entries(inputs).find(([, binding]) => {
        const reference = StepReferenceSchema.safeParse(binding);
        if (!reference.success) return false;
        const { step, element, product, where, checkpoint, variants } = reference.data;
        const pin = figurePin(where);
        return step === source.figure.step && element === source.figure.element &&
          product === source.figure.product && checkpoint == null && variants == null && pin !== undefined &&
          JSON.stringify(pin) === JSON.stringify(figurePin(source.figure.where));
      });
      const prefix = reportFigureRole(source);
      let role = existing?.[0] ?? prefix;
      if (!existing) {
        for (let suffix = 2; Object.prototype.hasOwnProperty.call(inputs, role); suffix += 1) role = prefix + '-' + suffix;
      }
      accepted = setBlocks(addFigure.label, 'dialog', [...report.blocks, { figure: role }], existing ? undefined : {
        ...inputs, [role]: source.figure,
      });
    }
    if (accepted) {
      setSelection({ scope, index: report.blocks.length });
      setDialog(undefined);
    }
  }

  function action(command: StudioCommand) {
    return <Button key={command.id} {...commandAttributes(command)}
      disabled={!command.enabled} onClick={command.invoke}>{studioName(command.name)}</Button>;
  }

  const body = report && <div className="mx-auto grid w-full max-w-3xl gap-4 p-4">
    {planned && <p role="status">{studioRunStatus(planned.status)}
      {planned.status === 'failed' && planned.reason
        && <span className="block text-shell-muted">{planned.reason}</span>}
    </p>}
    {planned && projection?.window_map?.problems.filter((problem) => problem.step === planned.key.step &&
      (problem.variant === null || problem.variant === planned.key.variant) && (problem.row == null || problem.row === planned.key.row))
      .map((problem, index) => <ProblemRow key={index} problem={problem} />)}
    {planned?.light?.changed && <ChangeRow change={planned.light.changed} />}
    {previewUrl && <SavedReportResult key={retained.job_id} url={previewUrl} />}
    <div className="flex flex-wrap items-end gap-3">
      <label className="grid min-w-36 flex-1 gap-1">{studioName('reportTitle')}
        <TextField defaultValue={report.title} key={scope + report.title} disabled={!editable}
          onBlur={({ currentTarget }) => {
            const title = currentTarget.value.trim() || report.title;
            currentTarget.value = title;
            if (title !== report.title) save(studioName('reportTitle'), 'title', [{ op: 'replace', path: '/payload/title', value: title }]);
          }}
          onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
        {refusalAt('title')}
      </label>
      <label className="grid gap-1">{studioName('completeness')}
        <Select aria-label={studioName('completeness')} disabled={!editable} value={report.completeness ?? 'complete'}
          items={[
            { value: 'complete', label: studioName('complete') },
            { value: 'allow_omissions', label: studioName('allowOmissions') },
          ]}
          onValueChange={(value) => save(studioName('completeness'), 'completeness', [{
            op: report.completeness === undefined ? 'add' : 'replace', path: '/payload/completeness', value,
          }])} />
        {refusalAt('completeness')}
      </label>
    </div>
    <ol className="grid gap-3" key={scope}>
      {report.blocks.map((block, index) => {
        const reference = 'figure' in block ? StepReferenceSchema.safeParse(inputs[block.figure]) : undefined;
        return <li key={index} className="grid gap-2 rounded-surface border border-shell-divider p-3"
          onFocus={() => setSelection({ scope, index })}>
          {'markdown' in block ? <label className="grid gap-1">{studioName('text')}
            <TextArea disabled={!editable} defaultValue={block.markdown} key={block.markdown}
              onBlur={({ currentTarget }) => {
                const markdown = currentTarget.value;
                if (!markdown.trim()) currentTarget.value = block.markdown;
                else if (markdown !== block.markdown) changeBlock(index, { markdown });
              }} />
          </label> : <>
            <span className="font-medium">{reference?.success && reference.data.element
              ? reference.data.step + ' · ' + reference.data.element : block.figure}</span>
            <label className="grid gap-1">{studioName('caption')}
              <TextField defaultValue={block.caption ?? ''} key={block.caption ?? ''} disabled={!editable}
                onBlur={({ currentTarget }) => {
                  const caption = currentTarget.value.trim() || null;
                  currentTarget.value = caption ?? '';
                  if (caption !== (block.caption ?? null)) changeBlock(index, { figure: block.figure, caption });
                }}
                onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
            </label>
          </>}
          {refusalAt(`block:${index}`)}
          <div className="flex flex-wrap gap-2">{blockActions.map((id) => action(blockCommand(id, index)))}</div>
        </li>;
      })}
    </ol>
    <div className="flex gap-2">{action(addText)}{action(addFigure)}</div>
  </div>;
  const overlay = opened && <Dialog ariaLabel={opened === 'text' ? addText.label : addFigure.label}
    manageFocus panelClassName="grid w-[min(28rem,90vw)] gap-3" onClose={() => setDialog(undefined)}>
    {refusalAt('dialog')}
    <form className="grid gap-2" onSubmit={addBlock}>
      {opened === 'text' ? <label className="grid gap-1">{studioName('text')}
        <TextArea name="block" required data-dialog-autofocus />
      </label> : <label className="grid gap-1">{studioName('figures')}
        <Select aria-label={studioName('figures')} value={figureChoice} onValueChange={setFigureChoice}
          items={figures.map((source) => ({ value: figureAddress(source), label: figureLabel(source) }))} />
      </label>}
      <Button type="submit" className="justify-self-end"
        disabled={!editable || opened === 'figure' && (!addFigure.enabled || figureChoice === null)}>
        {opened === 'text' ? addText.label : addFigure.label}
      </Button>
    </form>
  </Dialog>;
  return { body, commands, overlay };
}
