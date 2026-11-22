// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
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
vi.mock('@/features/projectFolder/SavedTrainWorkspace', async () => {
  const { ProjectWindow } = await import('@/features/projectWindow/ProjectWindow');
  const { defaultProjectWindowState, noViews } = await import('@/features/projectWindow/projectWindowState');
  return {
    default: ({ document }: { document: { configuration: { project: string } } }) => (
      <ProjectWindow state={defaultProjectWindowState} onStateChange={() => {}} projectName={document.configuration.project}
        views={noViews} renderView={() => ({ body: null })} map={null} mapDrawer={null} compactMap={null} commands={[]} />
    ),
  };
});

vi.stubGlobal('ResizeObserver', class { observe() {} unobserve() {} disconnect() {} });
HTMLDialogElement.prototype.showModal = function showModal() { this.open = true; };
HTMLDialogElement.prototype.close = function close() { this.open = false; };

const opened = (project: string, id: string) => ({
  configuration: { project, store: 'results' },
  snapshot: { project_id: id },
} as Awaited<ReturnType<typeof openProjectFolder>>);

afterEach(() => {
  cleanup();
  window.history.replaceState(null, '', '/');
});

it('opens once through an alias and binds the window to the returned project', async () => {
  window.history.replaceState(null, '', '/?project=archive&another=value');
  vi.mocked(openProjectFolder).mockResolvedValue(opened('Reaching', 'opened-project'));
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

it('keeps a palette opened while the project loads open once the project window replaces the loading one', async () => {
  let release!: () => void;
  vi.mocked(openProjectFolder).mockReturnValue(new Promise((resolve) => {
    release = () => resolve(opened('Reaching', 'held-project'));
  }));
  render(
    <QueryClientProvider client={new QueryClient()}>
      <App projectId="held-project" />
    </QueryClientProvider>,
  );
  fireEvent.keyDown(window, { key: 'k', code: 'KeyK', metaKey: true });
  expect(screen.getByRole('dialog', { name: 'Command Palette' })).toBeTruthy();
  release();
  expect(await screen.findByRole('heading', { name: 'Reaching', level: 1 })).toBeTruthy();
  expect(screen.getByRole('dialog', { name: 'Command Palette' })).toBeTruthy();
});
