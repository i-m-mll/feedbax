import { studioName } from '@/components/activity/studioVocabulary';
import { humanizeLabel } from '@/components/values/ValueSpecField';
import type { StudioProjectWindowMapProblem } from '@/generated/studioContracts';

type ProblemElement =
  | { kind: 'component'; id: string }
  | { kind: 'wire'; index: number }
  | { kind: 'taskParameter'; name: string };

export function problemReason({ path, detail }: StudioProjectWindowMapProblem) {
  const prefix = `${path}: `;
  return detail.startsWith(prefix) ? detail.slice(prefix.length) : detail;
}

export function problemTarget(problem: StudioProjectWindowMapProblem) {
  const pointer = problem.path.startsWith('/');
  const parts = pointer
    ? problem.path.slice(1).split('/').map((part) => part.replace(/~1/g, '/').replace(/~0/g, '~'))
    : problem.path.split('.');
  if (parts[0] === 'payload') parts.shift();
  const [field, group, id] = parts;
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
    readonly view: 'model' | 'task' | 'objective' | 'training' | 'runs';
    readonly label: string;
    readonly element?: ProblemElement;
  };
}
