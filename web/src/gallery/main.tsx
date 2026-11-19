import React from 'react';
import ReactDOM from 'react-dom/client';
import { Gallery } from './Gallery';
import '../index.css';
import { showScrollbarsWhileScrolling } from '@/components/ui/scrollbars';

showScrollbarsWhileScrolling();
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Gallery />
  </React.StrictMode>,
);
