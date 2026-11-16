// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { StrictMode, type PropsWithChildren } from 'react';
import { afterEach, expect, it, vi } from 'vitest';
import { openProjectFolder } from '@/api/projectFolders';
import App from './App';

const { bindJobs } = vi.hoisted(() => ({ bindJobs: vi.fn() }));
vi.mock('@/api/projectFolders', () => ({ openProjectFolder: vi.fn(), createProjectFolder: vi.fn() }));
vi.mock('@/hooks/useStudioJobs', () => ({
  StudioJobsProvider: ({ children, projectId }: PropsWithChildren<{ projectId: string }>) => {
    bindJobs(projectId);
    return children;
  },
}));
vi.mock('@/features/projectFolder/SavedTrainWorkspace', () => ({
  default: ({ document }: { document: { configuration: { project: string } } }) => (
    <main>{document.configuration.project}</main>
  ),
}));

afterEach(() => {
  cleanup();
  window.history.replaceState(null, '', '/');
});

it('opens once through an alias and binds the window to the returned project', async () => {
  window.history.replaceState(null, '', '/?project=archive&another=value');
  vi.mocked(openProjectFolder).mockResolvedValue({
    configuration: { project: 'Reaching', store: 'results' },
    snapshot: { project_id: 'opened-project' },
  } as Awaited<ReturnType<typeof openProjectFolder>>);
  render(
    <StrictMode>
      <QueryClientProvider client={new QueryClient()}>
        <App projectId="archive" />
      </QueryClientProvider>
    </StrictMode>,
  );
  expect(await screen.findByText('Reaching')).toBeTruthy();
  await waitFor(() => expect(window.location.search).toBe('?project=opened-project&another=value'));
  expect(openProjectFolder).toHaveBeenCalledExactlyOnceWith('archive');
  expect(new Set(bindJobs.mock.calls.flat())).toEqual(new Set(['opened-project']));
});
