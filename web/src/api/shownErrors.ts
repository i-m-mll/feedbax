import { studioReason } from '@/components/activity/studioVocabulary';
import { parseContract, type JsonValue, type StudioErrorReport } from '@/generated/studioContracts';
import { ApiRequestError, requestResponse } from './request';

/**
 * Every error Studio shows passes through here once: it gets a ref, its whole record goes to the console and the
 * service's log under that ref, and the element that shows its text carries the ref as `data-note-ref`.
 */
const shownText = new WeakMap<object, string>();
/** The ref of each recently shown text, newest last. */
const refs = new Map<string, string>();
const keptRefs = 100;

/** Studio's own defect: a request its check refused before sending, or an answer it could not read. */
export function isStudioDefect(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError && (error.kind === 'request' || error.kind === 'response');
}

/** The text an error is shown as, noted under its ref the first time it is shown. */
export function shownError(error: unknown, fallback = 'Request failed'): string {
  return noteError(error, (ref) => {
    if (error instanceof ApiRequestError && error.kind === 'request') return studioReason('studioDefect', ref);
    if (error instanceof ApiRequestError && error.kind === 'response') return studioReason('unreadableAnswer', ref);
    return error instanceof Error && error.message ? error.message : fallback;
  });
}

/** Notes a region's render failure, shown as `text`. */
export function shownRenderFailure(error: Error, region: string, componentStack: string, text: string) {
  return noteError(error, () => text, { kind: 'render', region: region.slice(0, 80), component_stack: componentStack.slice(0, 8192) });
}

function noteError(error: unknown, text: (ref: string) => string, extra: Partial<StudioErrorReport> = {}): string {
  const key = typeof error === 'object' && error !== null ? error : undefined;
  const known = key && shownText.get(key);
  if (known) return known;
  const ref = crypto.getRandomValues(new Uint32Array(1))[0].toString(16).padStart(8, '0');
  const shown = text(ref);
  if (key) shownText.set(key, shown);
  refs.delete(shown);
  refs.set(shown, ref);
  if (refs.size > keptRefs) refs.delete(refs.keys().next().value!);
  const report = errorReport(ref, error, extra);
  console.error(`Studio error ${ref}`, report, error);
  void requestResponse('/api/studio/errors', {
    method: 'POST', body: JSON.stringify(parseContract('StudioErrorReport', report)), keepalive: true,
  }).catch((failure: unknown) => console.error(`Studio error ${ref} could not reach the service log`, failure));
  return shown;
}

function json(value: unknown): JsonValue {
  if (value === undefined) return null;
  try {
    return JSON.parse(JSON.stringify(value)) as JsonValue;
  } catch {
    return String(value);
  }
}

function errorReport(ref: string, error: unknown, extra: Partial<StudioErrorReport>): StudioErrorReport {
  const failure = error instanceof Error ? error : new Error(String(error));
  const cause = (failure as Error & { cause?: unknown }).cause;
  const stacks = [failure.stack, cause instanceof Error ? `Caused by ${cause.stack}` : undefined].filter(Boolean).join('\n');
  const request = error instanceof ApiRequestError ? {
    kind: error.kind, path: error.path.slice(0, 2048), method: error.method, status: error.status,
    contract: error.violation?.contract, issues: error.violation?.issues.slice(0, 200).map((issue) => issue.slice(0, 1024)),
    sent: json(error.sent), detail: json(error.detail),
  } : { kind: 'error' as const };
  return { ref, message: failure.message.slice(0, 4096), stack: stacks.slice(0, 16384), ...request, ...extra };
}

/** The ref of the error an element shows, when it shows one. */
function refShown(text: string) {
  let found: string | undefined;
  for (const [shown, ref] of refs) if (text.includes(shown)) found = ref;
  return found;
}

function markAlert(element: Element) {
  const ref = refs.size ? refShown(element.textContent ?? '') : undefined;
  if (ref === undefined) element.removeAttribute('data-note-ref');
  else if (element.getAttribute('data-note-ref') !== ref) element.setAttribute('data-note-ref', ref);
}

const alerts = '[role="alert"]';

/** Keeps `data-note-ref` on every alert of `root` that shows a noted error. */
export function watchShownErrors(root: HTMLElement) {
  const observer = new MutationObserver((mutations) => {
    const touched = new Set<Element>();
    for (const { target, addedNodes } of mutations) {
      const element = target instanceof Element ? target : target.parentElement;
      const alert = element?.closest(alerts);
      if (alert) touched.add(alert);
      for (const node of addedNodes) {
        if (!(node instanceof Element)) continue;
        if (node.matches(alerts)) touched.add(node);
        node.querySelectorAll(alerts).forEach((inner) => touched.add(inner));
      }
    }
    touched.forEach(markAlert);
  });
  observer.observe(root, { subtree: true, childList: true, characterData: true });
  return () => observer.disconnect();
}
