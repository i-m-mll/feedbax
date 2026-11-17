import '@xyflow/react/dist/style.css';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { NodeStyles } from './NodeStyles';
import '../index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <NodeStyles />
  </React.StrictMode>,
);
