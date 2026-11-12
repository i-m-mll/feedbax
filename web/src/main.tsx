import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactFlowProvider } from '@xyflow/react';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import { StudioJobsProvider } from '@/hooks/useStudioJobs';
import App from './App';
import './index.css';
import '@xyflow/react/dist/style.css';

const queryClient = new QueryClient();
const folderProjectId = new URLSearchParams(window.location.search).get('project');

if (import.meta.env.VITE_FEEDBAX_E2E === '1') {
  void import('./test/e2eHarness').then(({ installE2EHarness }) => installE2EHarness());
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StudioErrorBoundary region="Studio">
      <QueryClientProvider client={queryClient}>
        <StudioJobsProvider folderProjectId={folderProjectId}>
          <ReactFlowProvider>
            <App projectId={folderProjectId} />
          </ReactFlowProvider>
        </StudioJobsProvider>
      </QueryClientProvider>
    </StudioErrorBoundary>
  </React.StrictMode>
);
