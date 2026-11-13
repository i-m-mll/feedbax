import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StudioErrorBoundary } from '@/components/layout/StudioErrorBoundary';
import App from './App';
import './index.css';
import '@xyflow/react/dist/style.css';

const queryClient = new QueryClient();
const query = new URLSearchParams(window.location.search);

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <StudioErrorBoundary region="Studio">
      <QueryClientProvider client={queryClient}>
        <App projectId={query.get('project')} activity={query.get('activity') === '1'} />
      </QueryClientProvider>
    </StudioErrorBoundary>
  </React.StrictMode>
);
