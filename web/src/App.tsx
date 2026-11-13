import { lazy, Suspense, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { createProjectFolder } from '@/api/projectFolders';
import { apiErrorMessage } from '@/api/request';
import { StudioJobsProvider } from '@/hooks/useStudioJobs';
import { StudioActivityPane } from '@/components/activity/StudioActivity';
import { studioName } from '@/components/activity/studioVocabulary';

const StudioProjectWorkspace = lazy(() => import('@/features/projectFolder/SavedTrainWorkspace'));

function Project({ id }: { id: string }) {
  return (
    <StudioJobsProvider projectId={id}>
      <Suspense fallback={<main aria-busy="true" />}>
        <StudioProjectWorkspace projectId={id} />
      </Suspense>
    </StudioJobsProvider>
  );
}

function NewProject() {
  const project = useQuery({
    queryKey: ['studio-entry-project'],
    queryFn: () => createProjectFolder(),
    retry: false,
    retryOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    staleTime: Infinity,
  });
  const id = project.data?.snapshot.project_id;
  useEffect(() => {
    if (!id) return;
    const url = new URL(window.location.href);
    url.searchParams.set('project', id);
    window.history.replaceState(window.history.state, '', url);
  }, [id]);

  if (id) return <Project id={id} />;
  if (project.isError && !project.isFetching) return (
    <main className="p-4">
      <p role="alert">{apiErrorMessage(project.error)}</p>
      <button type="button" onClick={() => void project.refetch()}>{studioName('retry')}</button>
    </main>
  );
  return <main aria-busy="true" />;
}

export default function App({ projectId, activity }: { projectId: string | null; activity?: boolean }) {
  if (activity) return (
    <StudioJobsProvider projectId={null}>
      <main className="h-screen overflow-auto bg-shell-surface p-3 text-shell-foreground">
        <h1 className="mb-2 font-medium">{studioName('activity')}</h1>
        <StudioActivityPane standalone projectNames={new Map()} onOpenRun={async () => false} onClose={() => {}} />
      </main>
    </StudioJobsProvider>
  );
  return projectId ? <Project id={projectId} /> : <NewProject />;
}
