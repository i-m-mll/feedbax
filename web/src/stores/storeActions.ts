import { toast } from 'sonner';

type ErrorToast = string | ((error: unknown) => string);

export interface StoreActionFeedbackOptions {
  errorToast: ErrorToast;
  toastId?: string;
  onError?: (error: unknown) => boolean | void;
}

export function actionErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }
  if (typeof error === 'string' && error.trim().length > 0) {
    return error;
  }
  return fallback;
}

export async function withStoreActionFeedback<T>(
  action: () => Promise<T>,
  options: StoreActionFeedbackOptions,
): Promise<T | undefined> {
  try {
    return await action();
  } catch (error) {
    const shouldReportError = options.onError?.(error);
    if (shouldReportError === false) return undefined;
    const message =
      typeof options.errorToast === 'function' ? options.errorToast(error) : options.errorToast;
    toast.error(message, options.toastId ? { id: options.toastId } : undefined);
    return undefined;
  }
}
