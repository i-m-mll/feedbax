export type ApiRequestErrorKind = 'network' | 'http' | 'contract';

export class ApiRequestError extends Error {
  readonly kind: ApiRequestErrorKind;
  readonly status: number | null;
  readonly path: string;

  constructor(
    kind: ApiRequestErrorKind,
    path: string,
    message: string,
    options: { status?: number | null; cause?: unknown } = {},
  ) {
    super(message);
    this.name = 'ApiRequestError';
    this.kind = kind;
    this.status = options.status ?? null;
    this.path = path;
    (this as Error & { cause?: unknown }).cause = options.cause;
  }
}

async function parseErrorBody(response: Response): Promise<string> {
  try {
    const text = await response.text();
    return text.trim();
  } catch {
    return '';
  }
}

export async function requestResponse(path: string, options?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(path, {
      headers: {
        'Content-Type': 'application/json',
        ...(options?.headers ?? {}),
      },
      ...options,
    });
  } catch (error) {
    throw new ApiRequestError(
      'network',
      path,
      'Backend unavailable. Check that the Feedbax Studio backend is running.',
      { cause: error },
    );
  }

  if (!response.ok) {
    const detail = await parseErrorBody(response);
    throw new ApiRequestError(
      'http',
      path,
      detail || `Request failed with status ${response.status}`,
      { status: response.status },
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

export function apiErrorMessage(error: unknown, fallback = 'Request failed'): string {
  if (error instanceof ApiRequestError) return error.message;
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}
