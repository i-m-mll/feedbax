import { serviceDidNotRespond, serviceResponded } from './connection';
import { ContractViolation, parseContract, type ContractName, type ContractTypeMap } from '@/generated/studioContracts';

/**
 * `request`: Studio's own check refused what it was about to send, so nothing was sent. `response`: the service's
 * answer did not satisfy its contract.
 */
export type ApiRequestErrorKind = 'network' | 'http' | 'request' | 'response';

export class ApiRequestError extends Error {
  readonly kind: ApiRequestErrorKind;
  readonly status: number | null;
  readonly path: string;
  readonly detail: unknown;
  readonly method: string | undefined;
  /** The body sent, or refused before sending. */
  readonly sent: unknown;

  constructor(
    kind: ApiRequestErrorKind,
    path: string,
    message: string,
    options: { status?: number | null; cause?: unknown; detail?: unknown; method?: string; sent?: unknown } = {},
  ) {
    super(message);
    this.name = 'ApiRequestError';
    this.kind = kind;
    this.status = options.status ?? null;
    this.path = path;
    this.detail = options.detail;
    this.method = options.method;
    this.sent = options.sent;
    (this as Error & { cause?: unknown }).cause = options.cause;
  }

  /** The contract violation behind a `request` or `response` error. */
  get violation(): ContractViolation | undefined {
    const { cause } = this as Error & { cause?: unknown };
    return cause instanceof ContractViolation ? cause : undefined;
  }
}

/** The JSON body of a request, once it satisfies `contract`; a body that does not is Studio's defect and is not sent. */
export function sentBody<K extends ContractName>(contract: K, body: ContractTypeMap[K], path: string, method = 'POST') {
  try {
    parseContract(contract, body);
  } catch (error) {
    throw new ApiRequestError('request', path, error instanceof Error ? error.message : String(error), {
      cause: error, method, sent: body,
    });
  }
  return JSON.stringify(body);
}

async function parseErrorBody(response: Response): Promise<unknown> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === 'object' && 'detail' in body) return body.detail;
  } catch {
    return undefined;
  }
}

function detailMessage(detail: unknown): string {
  if (typeof detail === 'string') return detail.trim();
  if (detail && typeof detail === 'object' && 'message' in detail &&
      typeof detail.message === 'string') {
    return detail.message.trim();
  }
  return '';
}

/** The service's response to a request; `answers` are statuses besides success whose body is the route's answer. */
export async function requestResponse(path: string, options?: RequestInit, answers: readonly number[] = []): Promise<Response> {
  let response: Response;
  const headers = new Headers(options?.headers);
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  try {
    response = await fetch(path, {
      ...options,
      headers,
    });
  } catch (error) {
    if (options?.signal?.aborted) throw error;
    serviceDidNotRespond();
    throw new ApiRequestError('network', path, 'Service unreachable', { cause: error, method: options?.method });
  }
  serviceResponded();

  if (!response.ok && !answers.includes(response.status)) {
    const detail = await parseErrorBody(response);
    throw new ApiRequestError(
      'http',
      path,
      detailMessage(detail) || `Request failed with status ${response.status}`,
      { status: response.status, detail, method: options?.method },
    );
  }

  return response;
}

export async function requestJson(path: string, options?: RequestInit): Promise<unknown> {
  const response = await requestResponse(path, options);
  return response.json() as Promise<unknown>;
}

/** A failure reading the service's answer at `path`: the error itself when the request failed, else a `response` error. */
export function asApiRequestError(error: unknown, path: string): ApiRequestError {
  if (error instanceof ApiRequestError) return error;
  return new ApiRequestError('response', path, error instanceof Error ? error.message : String(error), { cause: error });
}

/** The request got no response: the window's connection reports it, not the region that asked. */
export function isUnanswered(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError && error.kind === 'network';
}

/** The service did not answer, or failed answering: a fault of the connection or the service, not of the request. */
export function isServiceFailure(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError && (error.kind === 'network' || (error.status ?? 0) >= 500);
}

/** The service answered and refused the request for a reason of its own; the reason belongs on what was asked. */
export function isRefusal(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError && error.kind === 'http' && !isServiceFailure(error) && !isHttpConflict(error);
}

/** A failure of the gesture itself, shown on what it acted on: a refusal by the service or by the edit's own checks. A service failure or an outside change concerns the whole project. */
export function landsOnGesture(error: unknown): boolean {
  return !isServiceFailure(error) && !isHttpConflict(error);
}

export function isHttpConflict(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError && error.kind === 'http' && error.status === 409;
}
