import { lazy, Suspense, useLayoutEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { createProjectFolder, openProjectFolder } from '@/api/projectFolders';
import { apiErrorMessage } from '@/api/request';
import { StudioJobsProvider } from '@/hooks/useStudioJobs';
import { StudioActivityPane } from '@/components/activity/StudioActivity';
import { studioName } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { PendingProjectWindow, ProjectWindowPalette } from '@/features/projectWindow/ProjectWindow';

const StudioProjectWorkspace = lazy(() => import('@/features/projectFolder/SavedTrainWorkspace'));

function ProjectEntry({ requestedId }: { requestedId: string | null }) {
  const project = useQuery({
    queryKey: ['studio-entry-project', requestedId],
    queryFn: () => requestedId ? openProjectFolder(requestedId) : createProjectFolder(),
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    staleTime: Infinity,
  });
  const id = project.data?.snapshot.project_id;
  useLayoutEffect(() => {
    if (!id || id === requestedId) return;
    const url = new URL(window.location.href);
    url.searchParams.set('project', id);
    window.history.replaceState(window.history.state, '', url);
  }, [id, requestedId]);

  if (project.data) return (
    <StudioJobsProvider key={id} projectId={project.data.snapshot.project_id}>
      <Suspense fallback={<PendingProjectWindow />}>
        <StudioProjectWorkspace document={project.data} />
      </Suspense>
    </StudioJobsProvider>
  );
  if (project.isError && !project.isFetching) return (
    <main className="p-4">
      <p role="alert">{apiErrorMessage(project.error)}</p>
      <Button onClick={() => void project.refetch()}>{studioName('retry')}</Button>
    </main>
  );
  return <PendingProjectWindow />;
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
  return <ProjectWindowPalette><ProjectEntry requestedId={projectId} /></ProjectWindowPalette>;
}
