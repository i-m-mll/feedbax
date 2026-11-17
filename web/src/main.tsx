import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StudioRegionBoundary } from '@/components/activity/StudioRegion';
import App from './App';
import './index.css';
import '@xyflow/react/dist/style.css';

const queryClient = new QueryClient();
const query = new URLSearchParams(window.location.search);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StudioRegionBoundary name="studio" recoveryLabel="reload">
      <QueryClientProvider client={queryClient}>
        <App projectId={query.get('project')} activity={query.get('activity') === '1'} />
      </QueryClientProvider>
    </StudioRegionBoundary>
  </React.StrictMode>
);
