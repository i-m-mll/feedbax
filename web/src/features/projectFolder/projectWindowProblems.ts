import { studioName } from '@/components/activity/studioVocabulary';
import { humanizeLabel } from '@/components/values/ValueSpecField';
import type { JsonValue, StudioProjectFilePatch, StudioProjectWindowMapProblem } from '@/generated/studioContracts';
import { authoredMember, authoredPointerParts, removeAuthoredMember, type AuthoredProject } from './authoredEdits';

type ProblemElement =
  | { kind: 'component'; id: string }
  | { kind: 'analysisElement'; id: string }
  | { kind: 'wire'; index: number }
  | { kind: 'taskParameter'; name: string };

export function problemReason({ path, detail, document_path }: StudioProjectWindowMapProblem) {
  const prefix = `${path}: `;
  const message = detail.startsWith(prefix) ? detail.slice(prefix.length) : detail;
  return document_path ? `${document_path} · ${message}` : message;
}

const authoredGraphVariation = 'authored_graph_variation';

function isOverrideRecord(value: JsonValue | undefined): value is { path: string; op: 'add' | 'replace' | 'remove' } {
  return value != null && typeof value === 'object' && !Array.isArray(value) && typeof value.path === 'string' &&
    (value.op === 'add' || value.op === 'replace' || value.op === 'remove');
}

function scopedVariationWriter(path: readonly string[]): number | undefined {
  if (path[0] === 'deltas' && path[2] === 'patches') return 4;
  if (path[0] !== 'sweep') return;
  if (path[1] === 'axes' && path[3] === 'values' && path[5] === 'patches') return 7;
  if ((path[1] === 'exceptions' || path[1] === 'rows') && path[3] === 'patches') return 5;
}

function canRemoveWithoutReindexing(value: JsonValue | undefined, key: string) {
  return !Array.isArray(value) || Number(key) === value.length - 1;
}

export function removeGraphVariation(
  project: AuthoredProject, problem: StudioProjectWindowMapProblem,
): StudioProjectFilePatch | undefined {
  if (problem.code !== authoredGraphVariation || !problem.step ||
      problem.address !== `steps/${problem.step}.json` || !problem.authored_path ||
      project.steps?.[problem.step]?.kind !== 'evaluate') return;
  const address = { kind: 'step', id: problem.step } as const;
  const path = authoredPointerParts(problem.authored_path);
  const writerDepth = scopedVariationWriter(path);
  const member = (parts: readonly string[]) => authoredMember(project, address, parts);
  const recordPath = path.slice(0, -1);
  let target: string[];
  if (writerDepth !== undefined && path[writerDepth] === 'value') {
    const writer = path.slice(0, writerDepth);
    const suffix = path.slice(writerDepth + 1);
    const storedWriter = member(writer);
    const key = path[0] === 'deltas'
      ? (isOverrideRecord(storedWriter) ? storedWriter.path : undefined)
      : writer.at(-1);
    const writesRecord = /^params\.variation\.[^.]+$/.test(key ?? '');
    const writesPath = /^params\.variation\.[^.]+\.path$/.test(key ?? '');
    if (writesPath && suffix.length === 0 && typeof member(path) === 'string') {
      target = writer;
    } else if (writesRecord && suffix.length === 1 && suffix[0] === 'path' && isOverrideRecord(member(recordPath))) {
      if (!storedWriter || typeof storedWriter !== 'object' || Array.isArray(storedWriter) ||
          storedWriter.op !== 'add' || /^params\.variation\.[0-9]+$/.test(key ?? '')) return;
      target = writer;
    } else if (key === 'params.variation' && suffix.length === 2 && suffix[1] === 'path' &&
        isOverrideRecord(member(recordPath))) {
      if (!canRemoveWithoutReindexing(member(path.slice(0, writerDepth + 1)), suffix[0])) return;
      target = recordPath;
    } else return;
  } else if (path.length === 6 && path.slice(0, 4).join('/') === 'base/inline/params/variation' &&
      path.at(-1) === 'path' && isOverrideRecord(member(recordPath))) {
    if (!canRemoveWithoutReindexing(member(path.slice(0, 4)), path[4])) return;
    target = recordPath;
  } else return;
  const patch = removeAuthoredMember(project, address, target);
  return patch && { address, patches: [patch] };
}

export function problemTarget(problem: StudioProjectWindowMapProblem): ProblemReveal['target'] {
  if (problem.model_address || problem.code === authoredGraphVariation) return { view: 'model', label: studioName('model') };
  const pointer = problem.path.startsWith('/');
  const parts = pointer
    ? problem.path.slice(1).split('/').map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~'))
    : problem.path.split('.');
  if (parts[0] === 'payload') parts.shift();
  const [field, group, id] = parts;
  if (field === 'trial_documents') {
    if (group === 'model') return { view: 'model', label: parts[3] === 'components' ? studioName('components') : studioName('model') };
    if (group === 'task' || group === 'python_task') return { view: 'task', label: studioName('task') };
    if (group === 'objective') return { view: 'objective', label: studioName('objective') };
  }
  if (problem.element && (field === 'operations' || field === 'views')) {
    return {
      view: 'analysis', label: group ? humanizeLabel(group) : studioName('analysis'),
      element: { kind: 'analysisElement', id: problem.element },
    } as const;
  }
  if (field === 'graph') {
    if (group === 'nodes') {
      if (pointer && id) {
        return {
          view: 'model', label: parts[4] ? `${id} · ${humanizeLabel(parts[4])}` : id,
          element: { kind: 'component', id },
        } as const;
      }
      return { view: 'model', label: studioName('components') } as const;
    }
    if (group === 'wires') {
      const index = pointer && id && /^\d+$/.test(id) ? Number(id) : null;
      return {
        view: 'model', label: studioName('wires'),
        element: index === null ? undefined : { kind: 'wire', index },
      } as const;
    }
    return { view: 'model', label: studioName('model') } as const;
  }
  if (field === 'task' || field === 'task_binding_spec') {
    return {
      view: 'task', label: group === 'params' && id ? humanizeLabel(id) : studioName('task'),
      element: pointer && group === 'params' && id ? { kind: 'taskParameter', name: id } : undefined,
    } as const;
  }
  if (field === 'objective') return { view: 'objective', label: studioName('objective') } as const;
  if (field === 'training_config') {
    return { view: 'training', label: group ? humanizeLabel(group) : studioName('training') } as const;
  }
  if (field === 'method_ref' || field === 'method_payload') {
    return { view: 'training', label: studioName('trainingMethod') } as const;
  }
  const label = field === 'inputs' && group ? humanizeLabel(group)
    : problem.document ?? (field ? humanizeLabel(field) : studioName('step'));
  return { view: 'runs', label } as const;
}

export interface ProblemReveal {
  readonly problem: StudioProjectWindowMapProblem;
  readonly target: {
    readonly view: 'model' | 'task' | 'objective' | 'training' | 'analysis' | 'document' | 'runs';
    readonly label: string;
    readonly element?: ProblemElement;
  };
}
