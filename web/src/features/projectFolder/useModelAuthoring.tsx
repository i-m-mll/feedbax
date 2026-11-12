import { useRef, useState, type FormEvent } from 'react';
import { fetchDocumentAuthoring } from '@/api/projectFolders';
import { apiErrorMessage } from '@/api/request';
import { studioAccessibleName, studioName } from '@/components/activity/studioVocabulary';
import { Dialog } from '@/components/ui/Dialog';
import { bindStudioCommand, commandAttributes } from '@/features/projectWindow/studioCommands';
import {
  StudioModelAddComponentSchema,
  type ComponentDefinition, type GraphSpec, type PresentationPosition,
  type StudioModelAddComponent, type StudioModelAuthoringChange, type StudioSavedTrainKey,
} from '@/generated/studioContracts';
import { ProjectEditConflictError, type ProjectEditController } from './projectEditController';

type Operation = StudioModelAuthoringChange['operation'];
type Mode = Operation['kind'];
interface Review {
  revision: string;
  document: string;
  run: StudioSavedTrainKey;
  litSteps: readonly string[];
}

const control = 'rounded px-2 py-1 text-sm hover:bg-slate-100 disabled:opacity-50';
const panel = 'grid max-h-[75dvh] w-[min(28rem,90vw)] gap-3 overflow-auto rounded-shell-control border border-shell-divider bg-shell-chrome p-4 text-shell-foreground shadow-xl';

function ComponentChoices({ choices, enabled, choose }: {
  choices: readonly StudioModelAddComponent[];
  enabled: boolean;
  choose(operation: StudioModelAddComponent): void;
}) {
  return <ul className="space-y-1">{choices.map((operation) => {
    const label = operation.component.split('.').at(-1);
    return <li key={operation.component}>
      <button type="button" aria-label={studioAccessibleName('add', label)} disabled={!enabled}
        className={`${control} w-full text-left`} onClick={() => choose(operation)}>{label}</button>
    </li>;
  })}</ul>;
}

function ports(graph: GraphSpec, direction: 'input_ports' | 'output_ports') {
  return Object.entries(graph.nodes ?? {}).flatMap(([node, spec]) =>
    (spec[direction] ?? []).map((port) => ({ node, port, id: JSON.stringify([node, port]) })));
}

function PortForm({ mode, graph, commit, busy }: {
  mode: 'connectPorts' | 'exposeOutput';
  graph: GraphSpec;
  commit(operation: Operation): void;
  busy: boolean;
}) {
  const [sourceId, setSource] = useState('');
  const [targetId, setTarget] = useState('');
  const outputs = ports(graph, 'output_ports');
  const inputs = ports(graph, 'input_ports');
  const source = outputs.find(({ id }) => id === sourceId);
  const target = inputs.find(({ id }) => id === targetId);
  function submit(event: FormEvent) {
    event.preventDefault();
    if (!source || busy) return;
    if (mode === 'exposeOutput') {
      commit({ kind: mode, source_node: source.node, source_port: source.port });
    } else if (target) {
      commit({ kind: mode, wire: {
        source_node: source.node, source_port: source.port, target_node: target.node, target_port: target.port,
      } });
    }
  }
  return <form className="grid gap-3" onSubmit={submit}>
    <label className="grid gap-1 text-sm">{studioName('sourcePort')}
      <select data-dialog-autofocus value={sourceId} onChange={(event) => setSource(event.currentTarget.value)}>
        <option value="" disabled>{studioName('choose')}</option>
        {outputs.map(({ id, node, port }) => <option key={id} value={id}>{node}.{port}</option>)}
      </select>
    </label>
    {mode === 'connectPorts' && <label className="grid gap-1 text-sm">{studioName('targetPort')}
      <select value={targetId} onChange={(event) => setTarget(event.currentTarget.value)}>
        <option value="" disabled>{studioName('choose')}</option>
        {inputs.map(({ id, node, port }) => <option key={id} value={id}>{node}.{port}</option>)}
      </select>
    </label>}
    <button type="submit" className={control} disabled={busy || !source || mode === 'connectPorts' && !target}>
      {studioName(mode)}
    </button>
  </form>;
}

export function useModelAuthoring({
  editor, graph, registry, review, documentScope, onSelect, savePosition,
}: {
  editor: ProjectEditController;
  graph: GraphSpec | null;
  registry?: readonly ComponentDefinition[];
  review?: Review;
  documentScope?: { users: ReadonlySet<string>; lit: ReadonlySet<string> };
  onSelect(id: string): void;
  savePosition(document: string, node: string, position: PresentationPosition): Promise<unknown>;
}) {
  const scope = review && JSON.stringify(review);
  const current = useRef(scope);
  current.current = scope;
  const [opened, setOpened] = useState<{ scope: string; mode: Mode; point?: PresentationPosition }>();
  const [issue, setIssue] = useState<{ document: string; message: string }>();
  const [busy, setBusy] = useState(false);
  const committing = useRef(false);
  const cursor = useRef<{ scope: string; point: PresentationPosition }>();
  const [query, setQuery] = useState('');
  const dialog = opened?.scope === scope ? opened : undefined;
  const choices = (registry ?? []).flatMap(({ name, revision }) => {
    const parsed = StudioModelAddComponentSchema.safeParse({ kind: 'addComponent', component: name, revision });
    return parsed.success ? [parsed.data] : [];
  });
  const allLit = !!documentScope && documentScope.lit.size === documentScope.users.size;
  const enabled = !!scope && !!graph && allLit && !busy;
  const outputs = graph && ports(graph, 'output_ports').length;
  const inputs = graph && ports(graph, 'input_ports').length;
  const open = (mode: Mode, point?: PresentationPosition) => {
    if (!scope || !enabled) return;
    setIssue(undefined);
    setQuery('');
    setOpened({ scope, mode, point });
  };
  const commands = [
    bindStudioCommand('addComponent', () => open('addComponent', cursor.current?.scope === scope ? cursor.current?.point : undefined),
      { enabled: enabled && choices.length > 0 }),
    bindStudioCommand('connectPorts', () => open('connectPorts'), { enabled: enabled && !!outputs && !!inputs }),
    bindStudioCommand('exposeOutput', () => open('exposeOutput'), { enabled: enabled && !!outputs }),
  ];

  async function commit(operation: Operation, point?: PresentationPosition) {
    if (!review || !scope || !enabled || committing.current) return;
    committing.current = true;
    setBusy(true);
    setIssue(undefined);
    try {
      const { saved } = editor.getSnapshot();
      const result = await fetchDocumentAuthoring(saved.project_id, {
        revision: review.revision, run: review.run, field: 'graph',
        change: { lit_steps: [...review.litSteps], operation },
      });
      if (current.current !== scope || result.document !== review.document) throw new ProjectEditConflictError();
      editor.editFiles(studioName(operation.kind), result.files, { reviewedRevision: result.revision });
      if (result.created_component) onSelect(`component:${result.created_component}`);
      setOpened(undefined);
      await editor.flush();
      if (point && result.created_component) await savePosition(result.document, result.created_component, point);
    } catch (error) {
      setIssue({ document: review.document, message: apiErrorMessage(error) });
    } finally {
      committing.current = false;
      setBusy(false);
    }
  }

  const error = issue?.document === review?.document && issue ? <p role="alert" className="px-2 text-sm">{issue.message}</p> : null;
  return {
    commands,
    placing: Boolean(dialog?.point),
    issue: !dialog && error,
    rememberPoint: (point: PresentationPosition | null) => { cursor.current = scope && point ? { scope, point } : undefined; },
    openAt: enabled && choices.length ? (point: PresentationPosition) => open('addComponent', point) : undefined,
    drawer: choices.length > 0 && <section aria-label={studioName('addComponent')} className="space-y-1">
      <h3 className="px-2 text-xs font-medium">{studioName('addComponent')}</h3>
      <ComponentChoices choices={choices} enabled={enabled} choose={(operation) => { void commit(operation); }} />
      {commands.slice(1).map((command) => <button key={command.id} type="button" className={`${control} block`}
        {...commandAttributes(command)} disabled={!command.enabled} onClick={command.invoke}>{command.label}</button>)}
      {documentScope && !allLit && <p className="px-2 text-sm">Topology edits require all Model users to be lit.</p>}
    </section>,
    overlay: dialog && graph && <Dialog ariaLabel={studioName(dialog.mode)} onClose={() => setOpened(undefined)} manageFocus panelClassName={panel}>
      <h2 className="font-medium">{studioName(dialog.mode)}</h2>
      {dialog.mode === 'addComponent'
        ? <>
          <input data-dialog-autofocus aria-label={studioName('components')} value={query}
            onChange={(event) => setQuery(event.currentTarget.value)} className="rounded border border-shell-divider px-2 py-1" />
          <ComponentChoices choices={choices.filter(({ component }) => component.toLowerCase().includes(query.trim().toLowerCase()))}
            enabled={enabled} choose={(operation) => { void commit(operation, dialog.point); }} />
        </>
        : <PortForm mode={dialog.mode} graph={graph} busy={busy} commit={(operation) => { void commit(operation); }} />}
      {error}
    </Dialog>,
  };
}
