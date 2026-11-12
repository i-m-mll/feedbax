import { useState, type FormEvent } from 'react';
import { apiErrorMessage } from '@/api/request';
import { studioName } from '@/components/activity/studioVocabulary';
import { Dialog } from '@/components/ui/Dialog';
import { bindStudioCommand, commandAttributes, type StudioCommand, type StudioCommandId } from '@/features/projectWindow/studioCommands';
import type { PlanView } from '@/features/pipeline/planView';
import {
  ReportDocumentSchema, StepReferenceSchema, type ReportDocument, type StudioProjectFilePatch,
  type StudioSavedTrainRuns,
} from '@/generated/studioContracts';
import { isReportFigureSource, reportFigureRole, type ReportFigureSource } from './createManagedStep';
import type { ProjectEditController, ProjectEditSnapshot } from './projectEditController';

const control = 'rounded-shell-control border border-shell-divider px-2 py-1 text-sm disabled:opacity-50';
const button = control + ' hover:bg-shell-hover';
const label = 'grid gap-1 text-sm';
const dialogClass = 'grid w-[min(28rem,90vw)] gap-3 rounded-shell-control border border-shell-divider bg-shell-chrome p-3 text-shell-foreground shadow-xl';
type Block = ReportDocument['blocks'][number];
type BlockAction = Extract<StudioCommandId, 'moveEarlier' | 'moveLater' | 'remove'>;

function figureLabel(source: ReportFigureSource) {
  const { step, row } = source.key;
  return step + ' · ' + source.name + (row === 'default' ? '' : ' · ' + row);
}

function figureAddress({ key, name, figure }: ReportFigureSource) {
  return JSON.stringify([key.variant, key.step, key.row, name, figure.product]);
}

export function useManagedReport({
  editor, snapshot, plan, projection, variant, stepId, wholeStep, focused,
}: {
  editor: ProjectEditController;
  snapshot: ProjectEditSnapshot;
  plan: PlanView;
  projection?: StudioSavedTrainRuns;
  variant: string | null;
  stepId?: string;
  wholeStep: boolean;
  focused: boolean;
}) {
  const { draft, outside, pending, saved } = snapshot;
  const step = stepId ? draft.steps?.[stepId] : undefined;
  const base = step?.kind === 'report' ? step.base : undefined;
  const document = base && 'document' in base && !base.pointer ? base.document : undefined;
  const authored = document ? draft.documents?.[document] : undefined;
  const parsed = authored?.schema === 'feedbax.report_document.v1'
    ? ReportDocumentSchema.safeParse(authored.payload) : undefined;
  const report = parsed?.success ? parsed.data : undefined;
  const scope = JSON.stringify([saved.project_id, variant, stepId, document]);
  const [dialog, setDialog] = useState<{ scope: string; revision: string; kind: 'text' | 'figure' }>();
  const [selection, setSelection] = useState<{ scope: string; index: number }>();
  const [error, setError] = useState<{ scope: string; message: string }>();
  const shared = document && (plan.steps.some(({ id, documentUses }) => id !== stepId && documentUses?.includes(document))
    || Object.entries(draft.steps ?? {}).some(([id, { base }]) => id !== stepId && 'document' in base && base.document === document));
  const direct = !step?.sweep && !step?.deltas?.length && !Object.keys(step?.document_overrides ?? {}).length;
  const editable = !!report && !!document && wholeStep && variant === 'default' && direct &&
    projection?.revision === saved.revision && !draft.sweep && !pending && !outside && !shared;
  const figures = projection?.window_map?.analysis_elements?.filter(isReportFigureSource)
    .filter(({ key }) => key.variant === variant) ?? [];
  const inputs = step?.inputs ?? {};
  const opened = dialog?.scope === scope && dialog.revision === saved.revision && focused ? dialog.kind : undefined;
  const selected = selection?.scope === scope ? selection.index : undefined;

  function save(name: string, patches: StudioProjectFilePatch['patches'], nextInputs?: typeof inputs) {
    if (!editable || !stepId || !document) return false;
    const files: StudioProjectFilePatch[] = [{ address: { kind: 'document', id: document }, patches }];
    if (nextInputs) files.push({ address: { kind: 'step', id: stepId }, patches: [
      { op: step?.inputs ? 'replace' : 'add', path: '/inputs', value: nextInputs },
    ] });
    try {
      editor.editFiles(name, files, { reviewedRevision: saved.revision });
      setError(undefined);
      return true;
    } catch (cause) {
      setError({ scope, message: apiErrorMessage(cause) });
      return false;
    }
  }

  function setBlocks(name: string, blocks: Block[], nextInputs?: typeof inputs) {
    return save(name, [{ op: 'replace', path: '/payload/blocks', value: blocks }], nextInputs);
  }

  function changeBlock(index: number, block: Block) {
    const previous = report?.blocks[index];
    if (!previous) return;
    save('markdown' in block ? studioName('text') : studioName('caption'), [{
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
        if (setBlocks(studioName(id), reordered)) setSelection({ scope, index: target });
        return;
      }
      const remaining = blocks.filter((_, position) => position !== index);
      let nextInputs: typeof inputs | undefined;
      if ('figure' in block && Object.prototype.hasOwnProperty.call(inputs, block.figure) &&
          !remaining.some((item) => 'figure' in item && item.figure === block.figure)) {
        nextInputs = { ...inputs };
        delete nextInputs[block.figure];
      }
      if (setBlocks(studioName(id), remaining, nextInputs)) {
        setSelection({ scope, index: Math.min(index, remaining.length - 1) });
      }
    }, {
      enabled: editable && !!block && (id === 'remove' ? blocks.length > 1 : target >= 0 && target < blocks.length),
      subject: block ? String(index + 1) : undefined,
    });
  }

  const blockActions = ['moveEarlier', 'moveLater', 'remove'] as const;
  const addText = bindStudioCommand('addText', () => setDialog({ scope, revision: saved.revision, kind: 'text' }), { enabled: editable });
  const addFigure = bindStudioCommand('addFigure', () => setDialog({ scope, revision: saved.revision, kind: 'figure' }), {
    enabled: editable && figures.length > 0,
  });
  const commands = focused && report ? [
    addText, addFigure,
    ...(selected === undefined ? [] : blockActions.map((id) => blockCommand(id, selected))),
  ] : [];

  function addBlock(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!opened || !report) return;
    const value = new FormData(event.currentTarget).get('block');
    if (typeof value !== 'string' || !value.trim()) return;
    let accepted: boolean;
    if (opened === 'text') {
      accepted = setBlocks(addText.label, [...report.blocks, { markdown: value }]);
    } else {
      const source = figures.find((figure) => figureAddress(figure) === value);
      if (!source) return;
      const existing = Object.entries(inputs).find(([, binding]) => {
        const reference = StepReferenceSchema.safeParse(binding);
        if (!reference.success) return false;
        const { step, row, element, product, where, checkpoint, variants } = reference.data;
        return step === source.figure.step && row === source.figure.row && element === source.figure.element &&
          product === source.figure.product && where == null && checkpoint == null && variants == null;
      });
      const prefix = reportFigureRole(source);
      let role = existing?.[0] ?? prefix;
      if (!existing) {
        for (let suffix = 2; Object.prototype.hasOwnProperty.call(inputs, role); suffix += 1) role = prefix + '-' + suffix;
      }
      accepted = setBlocks(addFigure.label, [...report.blocks, { figure: role }], existing ? undefined : {
        ...inputs, [role]: source.figure,
      });
    }
    if (accepted) {
      setSelection({ scope, index: report.blocks.length });
      setDialog(undefined);
    }
  }

  function action(command: StudioCommand) {
    return <button key={command.id} type="button" className={button} {...commandAttributes(command)}
      disabled={!command.enabled} onClick={command.invoke}>{studioName(command.name)}</button>;
  }

  const issue = error?.scope === scope ? <p role="alert">{error.message}</p> : null;
  const body = report && <div className="mx-auto grid w-full max-w-3xl gap-4 p-4">
    {!opened && issue}
    <div className="flex flex-wrap items-end gap-3">
      <label className={label + ' min-w-36 flex-1'}>{studioName('reportTitle')}
        <input className={control} defaultValue={report.title} key={scope + report.title} disabled={!editable}
          onBlur={({ currentTarget }) => {
            const title = currentTarget.value.trim() || report.title;
            currentTarget.value = title;
            if (title !== report.title) save(studioName('reportTitle'), [{ op: 'replace', path: '/payload/title', value: title }]);
          }}
          onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
      </label>
      <label className={label}>{studioName('completeness')}
        <select className={control} disabled={!editable} value={report.completeness ?? 'complete'}
          onChange={({ currentTarget }) => save(studioName('completeness'), [{
            op: report.completeness === undefined ? 'add' : 'replace', path: '/payload/completeness',
            value: currentTarget.value,
          }])}>
          <option value="complete">{studioName('complete')}</option>
          <option value="allow_omissions">{studioName('allowOmissions')}</option>
        </select>
      </label>
    </div>
    <ol className="grid gap-3" key={scope}>
      {report.blocks.map((block, index) => {
        const reference = 'figure' in block ? StepReferenceSchema.safeParse(inputs[block.figure]) : undefined;
        return <li key={index} className="grid gap-2 rounded-shell-control border border-shell-divider p-3"
          onFocus={() => setSelection({ scope, index })}>
          {'markdown' in block ? <label className={label}>{studioName('text')}
            <textarea className={control + ' min-h-20'} disabled={!editable} defaultValue={block.markdown} key={block.markdown}
              onBlur={({ currentTarget }) => {
                const markdown = currentTarget.value;
                if (!markdown.trim()) currentTarget.value = block.markdown;
                else if (markdown !== block.markdown) changeBlock(index, { markdown });
              }} />
          </label> : <>
            <span className="text-sm font-medium">{reference?.success && reference.data.element
              ? reference.data.step + ' · ' + reference.data.element : block.figure}</span>
            <label className={label}>{studioName('caption')}
              <input className={control} defaultValue={block.caption ?? ''} key={block.caption ?? ''} disabled={!editable}
                onBlur={({ currentTarget }) => {
                  const caption = currentTarget.value.trim() || null;
                  currentTarget.value = caption ?? '';
                  if (caption !== (block.caption ?? null)) changeBlock(index, { figure: block.figure, caption });
                }}
                onKeyDown={(event) => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
            </label>
          </>}
          <div className="flex flex-wrap gap-2">{blockActions.map((id) => action(blockCommand(id, index)))}</div>
        </li>;
      })}
    </ol>
    <div className="flex gap-2">{action(addText)}{action(addFigure)}</div>
  </div>;
  const overlay = opened && <Dialog ariaLabel={opened === 'text' ? addText.label : addFigure.label}
    manageFocus panelClassName={dialogClass} onClose={() => setDialog(undefined)}>
    {issue}
    <form className="grid gap-2" onSubmit={addBlock}>
      {opened === 'text' ? <label className={label}>{studioName('text')}
        <textarea className={control + ' min-h-24'} name="block" required data-dialog-autofocus />
      </label> : <label className={label}>{studioName('figures')}
        <select className={control} name="block" defaultValue="" required data-dialog-autofocus>
          <option value="" disabled />
          {figures.map((source) => {
            const address = figureAddress(source);
            return <option key={address} value={address}>{figureLabel(source)}</option>;
          })}
        </select>
      </label>}
      <button className={button} type="submit" disabled={!editable || opened === 'figure' && !addFigure.enabled}>
        {opened === 'text' ? addText.label : addFigure.label}
      </button>
    </form>
  </Dialog>;
  return { body, commands, overlay };
}
