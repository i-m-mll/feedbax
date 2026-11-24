import React from 'react';
import ReactDOM from 'react-dom/client';
import { Gallery } from './Gallery';
import { TaskViewGallery } from './TaskViewGallery';
import '../index.css';
import { showScrollbarsWhileScrolling } from '@/components/ui/scrollbars';

/** `gallery.html#task` draws the Task view's mockup; otherwise the controls. */
function Page() {
  const [page, setPage] = React.useState(location.hash);
  React.useEffect(() => {
    const follow = () => setPage(location.hash);
    window.addEventListener('hashchange', follow);
    return () => window.removeEventListener('hashchange', follow);
  }, []);
  return page === '#task' ? <TaskViewGallery /> : <Gallery />;
}

showScrollbarsWhileScrolling();
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Page />
  </React.StrictMode>,
);
