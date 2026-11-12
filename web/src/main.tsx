import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import App from './App';
import './index.css';
import '@xyflow/react/dist/style.css';

const queryClient = new QueryClient();
const folderProjectId = new URLSearchParams(window.location.search).get('project');

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StudioErrorBoundary region="Studio">
      <QueryClientProvider client={queryClient}>
        <App projectId={folderProjectId} />
      </QueryClientProvider>
    </StudioErrorBoundary>
  </React.StrictMode>
);
