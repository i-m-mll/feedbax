/**
 * Loss term operations for manipulating loss specifications.
 */

import type { LossTermSpec, TimeAggregationSpec } from '@/types/training';

/**
 * Get a loss term at a given path.
 */
export function getLossTermAtPath(
  root: LossTermSpec,
  path: string[]
): LossTermSpec | null {
  let current: LossTermSpec | undefined = root;
  for (const key of path) {
    current = current?.children?.[key];
    if (!current) return null;
  }
  return current;
}

/**
 * Update a loss term at a given path with partial updates.
 */
export function updateLossTermAtPath(
  root: LossTermSpec,
  path: string[],
  updates: Partial<LossTermSpec>
): LossTermSpec {
  if (path.length === 0) {
    return { ...root, ...updates };
  }
  if (!root.children) return root;
  const [head, ...rest] = path;
  const child = root.children[head];
  if (!child) return root;
  return {
    ...root,
    children: {
      ...root.children,
      [head]: updateLossTermAtPath(child, rest, updates),
    },
  };
}

/**
 * Add a new loss term at a given parent path.
 */
export function addLossTermAtPath(
  root: LossTermSpec,
  parentPath: string[],
  key: string,
  term: LossTermSpec
): LossTermSpec {
  if (parentPath.length === 0) {
    return {
      ...root,
      children: {
        ...(root.children ?? {}),
        [key]: term,
      },
    };
  }
  if (!root.children) return root;
  const [head, ...rest] = parentPath;
  const child = root.children[head];
  if (!child) return root;
  return {
    ...root,
    children: {
      ...root.children,
      [head]: addLossTermAtPath(child, rest, key, term),
    },
  };
}

/**
 * Remove a loss term at a given path.
 */
export function removeLossTermAtPath(
  root: LossTermSpec,
  path: string[]
): LossTermSpec {
  if (path.length === 0) {
    // Cannot remove root
    return root;
  }
  if (path.length === 1) {
    if (!root.children) return root;
    const { [path[0]]: _removed, ...remaining } = root.children;
    return {
      ...root,
      children: Object.keys(remaining).length > 0 ? remaining : undefined,
    };
  }
  if (!root.children) return root;
  const [head, ...rest] = path;
  const child = root.children[head];
  if (!child) return root;
  return {
    ...root,
    children: {
      ...root.children,
      [head]: removeLossTermAtPath(child, rest),
    },
  };
}

/**
 * Collect all leaf loss terms with their paths.
 */
export interface LossTermWithPath {
  path: string[];
  term: LossTermSpec;
  effectiveWeight: number;
}

export function collectLeafTerms(root: LossTermSpec): LossTermWithPath[] {
  const result: LossTermWithPath[] = [];

  const walk = (term: LossTermSpec, path: string[], weightScale: number) => {
    const effectiveWeight = weightScale * term.weight;
    const children = term.children ? Object.entries(term.children) : [];

    if (children.length === 0) {
      result.push({ path, term, effectiveWeight });
    } else {
      for (const [key, child] of children) {
        walk(child, [...path, key], effectiveWeight);
      }
    }
  };

  walk(root, [], 1);
  return result;
}

/**
 * Count total loss terms in a specification.
 */
export function countLossTerms(root: LossTermSpec): number {
  let count = 1;
  if (root.children) {
    for (const child of Object.values(root.children)) {
      count += countLossTerms(child);
    }
  }
  return count;
}

/**
 * Generate a unique key for a new loss term.
 */
export function generateUniqueKey(root: LossTermSpec, baseKey: string): string {
  const existingKeys = new Set<string>();

  const collectKeys = (term: LossTermSpec) => {
    if (term.children) {
      for (const key of Object.keys(term.children)) {
        existingKeys.add(key);
        collectKeys(term.children[key]);
      }
    }
  };

  collectKeys(root);

  let key = baseKey;
  let counter = 1;
  while (existingKeys.has(key)) {
    key = `${baseKey}_${counter}`;
    counter++;
  }
  return key;
}

/**
 * Create a default loss term specification.
 */
export function createDefaultLossTerm(
  type: string = 'TargetStateLoss',
  label: string = 'New Loss Term'
): LossTermSpec {
  return {
    type,
    label,
    weight: 1.0,
    norm: 'squared_l2',
    time_agg: {
      mode: 'all',
    },
  };
}

/**
 * Clone a loss term specification (deep copy).
 */
export function cloneLossTerm(term: LossTermSpec): LossTermSpec {
  const clone: LossTermSpec = {
    type: term.type,
    label: term.label,
    weight: term.weight,
  };

  if (term.selector) {
    clone.selector = term.selector;
  }

  if (term.norm) {
    clone.norm = term.norm;
  }

  if (term.time_agg) {
    clone.time_agg = { ...term.time_agg };
    if (term.time_agg.time_idxs) {
      clone.time_agg.time_idxs = [...term.time_agg.time_idxs];
    }
  }

  if (term.children) {
    clone.children = {};
    for (const [key, child] of Object.entries(term.children)) {
      clone.children[key] = cloneLossTerm(child);
    }
  }

  return clone;
}

/**
 * Get all selectors used in a loss specification.
 */
export function collectSelectors(root: LossTermSpec): string[] {
  const selectors: string[] = [];

  const walk = (term: LossTermSpec) => {
    if (term.selector) {
      selectors.push(term.selector);
    }
    if (term.children) {
      for (const child of Object.values(term.children)) {
        walk(child);
      }
    }
  };

  walk(root);
  return selectors;
}
