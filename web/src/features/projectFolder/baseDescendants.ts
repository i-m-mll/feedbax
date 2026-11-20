import type { Project } from '@/generated/studioContracts';

/**
 * `steps` and the steps based on one of them through `base: {step}` at any depth. A step based on another reads everything
 * stored on it, so an edit stored on a step reaches these, and they are lit whenever it is.
 */
export function withBaseDescendants(project: Pick<Project, 'steps'>, steps: Iterable<string>): Set<string> {
  const found = new Set(steps);
  for (let size = -1; size !== found.size;) {
    size = found.size;
    for (const [id, step] of Object.entries(project.steps ?? {})) {
      if (step && 'step' in step.base && found.has(step.base.step)) found.add(id);
    }
  }
  return found;
}
