import { describe, expect, it, vi } from 'vitest';
import { actionErrorMessage, withStoreActionFeedback } from '@/stores/storeActions';
import { toast } from 'sonner';

vi.mock('sonner', () => ({
  toast: {
    error: vi.fn(),
  },
}));

describe('store action feedback', () => {
  it('uses useful error messages before falling back', () => {
    expect(actionErrorMessage(new Error('Backend unavailable'), 'Fallback')).toBe(
      'Backend unavailable'
    );
    expect(actionErrorMessage('Plain failure', 'Fallback')).toBe('Plain failure');
    expect(actionErrorMessage(null, 'Fallback')).toBe('Fallback');
  });

  it('surfaces failed async actions through Sonner and returns undefined', async () => {
    const onError = vi.fn();
    const result = await withStoreActionFeedback(
      async () => {
        throw new Error('Network down');
      },
      {
        errorToast: (error) => actionErrorMessage(error, 'Failed'),
        toastId: 'store-action-test',
        onError,
      },
    );

    expect(result).toBeUndefined();
    expect(onError).toHaveBeenCalledWith(expect.any(Error));
    expect(toast.error).toHaveBeenCalledWith('Network down', { id: 'store-action-test' });
  });
});
