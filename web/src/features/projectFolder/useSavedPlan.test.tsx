// @vitest-environment jsdom

import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';
import { expect, it, vi } from 'vitest';
import type { StudioProjectSnapshot, StudioSavedTrainRuns } from '@/generated/studioContracts';
import { ProjectEditController, type ProjectEditTransport } from './projectEditController';
import { useSavedPlan } from './useSavedPlan';

const fetchSavedTrainRuns = vi.hoisted(() => vi.fn());
vi.mock('@/api/jobs', () => ({ fetchSavedTrainRuns }));

const hex = (number: number) => number.toString(16).padStart(64, '0');
const snapshot: StudioProjectSnapshot = {
  names: { documents: {}, steps: {}, elements: {} }, variant_names: {},
  project_id: 'reach', revision: hex(1), presentation_revision: hex(2), semantic_root: hex(3), presentation: null,
  declaration: { revision: hex(3), data: btoa('{}'), authored_revision: hex(3) }, file_revisions: [], project: {},
};

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

// Catches: a run's status shown in the window stays as first read because the plan is never read again.
it('reads the saved plan again when the Service says it changed, and only then', async () => {
  let status = 'queued';
  fetchSavedTrainRuns.mockImplementation(async () => ({ revision: hex(1), rows: [], status } as unknown as StudioSavedTrainRuns));
  const editor = new ProjectEditController({ configuration: { project: 'reach', store: 'results' }, snapshot },
    {} as ProjectEditTransport);
  const { result } = renderHook(() => useSavedPlan(editor), { wrapper });
  await waitFor(() => expect(result.current.current).toBe(true));
  expect(fetchSavedTrainRuns).toHaveBeenCalledTimes(1);

  status = 'running';
  act(() => editor.planChanged());
  await waitFor(() => expect((result.current.shown as unknown as { status: string }).status).toBe('running'));
  expect(fetchSavedTrainRuns).toHaveBeenCalledTimes(2);
  editor.dispose();
});
