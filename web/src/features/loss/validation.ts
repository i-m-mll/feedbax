/**
 * Client-side validation for loss specifications.
 */

import type { LossTermSpec, TimeAggregationSpec, NormFunction } from '@/types/training';

export interface ValidationError {
  path: string[];
  field: string;
  message: string;
}

const VALID_NORMS: NormFunction[] = ['squared_l2', 'l2', 'l1', 'huber'];

const VALID_LOSS_TYPES = [
  'Composite',
  'TargetStateLoss',
  'EffortLoss',
  'RegularizationLoss',
];

/**
 * Validate a loss term specification client-side.
 * Returns a list of validation errors.
 */
export function validateLossTerm(term: LossTermSpec): ValidationError[] {
  const errors: ValidationError[] = [];
  validateTermRecursive(term, [], errors);
  return errors;
}

function validateTermRecursive(
  term: LossTermSpec,
  path: string[],
  errors: ValidationError[]
): void {
  // Validate type
  if (!term.type) {
    errors.push({
      path,
      field: 'type',
      message: 'Loss type is required',
    });
  } else if (!VALID_LOSS_TYPES.includes(term.type)) {
    errors.push({
      path,
      field: 'type',
      message: `Unknown loss type: ${term.type}`,
    });
  }

  // Validate label
  if (!term.label || term.label.trim().length === 0) {
    errors.push({
      path,
      field: 'label',
      message: 'Label is required',
    });
  }

  // Validate weight
  if (typeof term.weight !== 'number' || !Number.isFinite(term.weight)) {
    errors.push({
      path,
      field: 'weight',
      message: 'Weight must be a finite number',
    });
  } else if (term.weight < 0) {
    errors.push({
      path,
      field: 'weight',
      message: 'Weight must be non-negative',
    });
  }

  // Validate norm (if present)
  if (term.norm && !VALID_NORMS.includes(term.norm)) {
    errors.push({
      path,
      field: 'norm',
      message: `Unknown norm function: ${term.norm}`,
    });
  }

  // Validate time aggregation (if present)
  if (term.time_agg) {
    validateTimeAggregation(term.time_agg, path, errors);
  }

  // Validate selector format (if present)
  if (term.selector) {
    validateSelector(term.selector, path, errors);
  }

  // For non-composite terms, selector is typically required
  const hasChildren = term.children && Object.keys(term.children).length > 0;
  if (!hasChildren && term.type !== 'Composite' && !term.selector) {
    errors.push({
      path,
      field: 'selector',
      message: 'Selector is required for non-composite loss terms',
    });
  }

  // Recursively validate children
  if (term.children) {
    for (const [key, child] of Object.entries(term.children)) {
      validateTermRecursive(child, [...path, key], errors);
    }
  }
}

function validateTimeAggregation(
  timeAgg: TimeAggregationSpec,
  path: string[],
  errors: ValidationError[]
): void {
  const validModes = ['all', 'final', 'range', 'segment', 'custom'];
  if (!validModes.includes(timeAgg.mode)) {
    errors.push({
      path,
      field: 'time_agg.mode',
      message: `Unknown time aggregation mode: ${timeAgg.mode}`,
    });
  }

  if (timeAgg.mode === 'range') {
    if (timeAgg.start === undefined || timeAgg.start === null) {
      errors.push({
        path,
        field: 'time_agg.start',
        message: 'Start index is required for range mode',
      });
    }
    if (timeAgg.end === undefined || timeAgg.end === null) {
      errors.push({
        path,
        field: 'time_agg.end',
        message: 'End index is required for range mode',
      });
    }
    if (
      timeAgg.start !== undefined &&
      timeAgg.end !== undefined &&
      timeAgg.start > timeAgg.end
    ) {
      errors.push({
        path,
        field: 'time_agg',
        message: 'Start index must be less than or equal to end index',
      });
    }
  }

  if (timeAgg.mode === 'segment') {
    if (!timeAgg.segment_name || timeAgg.segment_name.trim().length === 0) {
      errors.push({
        path,
        field: 'time_agg.segment_name',
        message: 'Segment name is required for segment mode',
      });
    }
  }

  if (timeAgg.mode === 'custom') {
    if (!timeAgg.time_idxs || timeAgg.time_idxs.length === 0) {
      errors.push({
        path,
        field: 'time_agg.time_idxs',
        message: 'Time indices are required for custom mode',
      });
    }
  }

  if (timeAgg.discount === 'power') {
    if (timeAgg.discount_exp === undefined || timeAgg.discount_exp === null) {
      errors.push({
        path,
        field: 'time_agg.discount_exp',
        message: 'Discount exponent is required for power discount',
      });
    } else if (timeAgg.discount_exp < 0) {
      errors.push({
        path,
        field: 'time_agg.discount_exp',
        message: 'Discount exponent must be non-negative',
      });
    }
  }
}

function validateSelector(
  selector: string,
  path: string[],
  errors: ValidationError[]
): void {
  // Check selector format
  const validPrefixes = ['probe:', 'port:', 'path:'];
  const hasValidPrefix = validPrefixes.some((prefix) => selector.startsWith(prefix));

  if (!hasValidPrefix) {
    errors.push({
      path,
      field: 'selector',
      message: `Selector must start with one of: ${validPrefixes.join(', ')}`,
    });
    return;
  }

  // Validate port selector format
  if (selector.startsWith('port:')) {
    const portRef = selector.slice(5);
    if (!portRef.includes('.')) {
      errors.push({
        path,
        field: 'selector',
        message: 'Port selector must be in format "port:node.port"',
      });
    }
  }

  // Validate probe selector format
  if (selector.startsWith('probe:')) {
    const probeId = selector.slice(6);
    if (probeId.length === 0) {
      errors.push({
        path,
        field: 'selector',
        message: 'Probe selector must specify a probe ID',
      });
    }
  }

  // Validate path selector format
  if (selector.startsWith('path:')) {
    const pathStr = selector.slice(5);
    if (pathStr.length === 0) {
      errors.push({
        path,
        field: 'selector',
        message: 'Path selector must specify a path',
      });
    }
  }
}

/**
 * Check if a loss specification is valid (no errors).
 */
export function isValidLossTerm(term: LossTermSpec): boolean {
  return validateLossTerm(term).length === 0;
}

/**
 * Get errors for a specific path in the loss tree.
 */
export function getErrorsAtPath(
  errors: ValidationError[],
  path: string[]
): ValidationError[] {
  const pathStr = path.join('/');
  return errors.filter((e) => e.path.join('/') === pathStr);
}

/**
 * Check if there are errors at or below a given path.
 */
export function hasErrorsAtOrBelowPath(
  errors: ValidationError[],
  path: string[]
): boolean {
  const pathStr = path.join('/');
  return errors.some(
    (e) => e.path.join('/') === pathStr || e.path.join('/').startsWith(pathStr + '/')
  );
}
