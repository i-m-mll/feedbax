import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { createRoot } from 'react-dom/client';
import { StudioPipelineMap } from '@/components/pipeline/StudioPipelineMap';
import SavedTrainWorkspace from '@/features/projectFolder/SavedTrainWorkspace';
import { StudioJobsProvider } from '@/hooks/useStudioJobs';
import '@/index.css';

const project = new URLSearchParams(window.location.search).get('project');
if (!project) throw new Error('The saved Train witness requires a project.');

createRoot(document.getElementById('root')!).render(
  <QueryClientProvider client={new QueryClient()}>
    <StudioJobsProvider folderProjectId={project}>
      <SavedTrainWorkspace projectId={project} placeRegions={({ map, details }) => (
        <>
          <StudioPipelineMap {...map} />
          {details}
        </>
      )} />
    </StudioJobsProvider>
  </QueryClientProvider>,
);
