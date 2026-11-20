import { serviceDidNotRespond, serviceResponded } from './connection';

export type ApiRequestErrorKind = 'network' | 'http' | 'contract';

export class ApiRequestError extends Error {
  readonly kind: ApiRequestErrorKind;
  readonly status: number | null;
  readonly path: string;
  readonly detail: unknown;

  constructor(
    kind: ApiRequestErrorKind,
    path: string,
    message: string,
    options: { status?: number | null; cause?: unknown; detail?: unknown } = {},
  ) {
    super(message);
    this.name = 'ApiRequestError';
    this.kind = kind;
    this.status = options.status ?? null;
    this.path = path;
    this.detail = options.detail;
    (this as Error & { cause?: unknown }).cause = options.cause;
  }
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

export async function requestResponse(path: string, options?: RequestInit): Promise<Response> {
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
    throw new ApiRequestError(
      'network',
      path,
      'Service unreachable',
      { cause: error },
    );
  }
  serviceResponded();

  if (!response.ok) {
    const detail = await parseErrorBody(response);
    throw new ApiRequestError(
      'http',
      path,
      detailMessage(detail) || `Request failed with status ${response.status}`,
      { status: response.status, detail },
    );
  }

  return response;
}

export async function requestJson(path: string, options?: RequestInit): Promise<unknown> {
  const response = await requestResponse(path, options);
  return response.json() as Promise<unknown>;
}

export function asApiRequestError(error: unknown, path: string, context: string): ApiRequestError {
  if (error instanceof ApiRequestError) return error;
  return new ApiRequestError('contract', path, context, { cause: error });
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

export function isHttpConflict(error: unknown): error is ApiRequestError {
  return error instanceof ApiRequestError && error.kind === 'http' && error.status === 409;
}

export function apiErrorMessage(error: unknown, fallback = 'Request failed'): string {
  if (error instanceof ApiRequestError) return error.message;
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
