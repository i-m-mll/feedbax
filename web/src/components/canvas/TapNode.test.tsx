// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { ReactFlowProvider, type NodeProps } from '@xyflow/react';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { TapNode } from './TapNode';
import type { TapNodeData } from '@/types/graph';

function renderTapNode(data: TapNodeData, selected = false) {
  const props = {
    id: 'tap:probe-1',
    data,
    selected,
    type: 'tap',
    deletable: true,
    selectable: true,
    draggable: true,
    dragging: false,
    zIndex: 0,
    isConnectable: true,
    positionAbsoluteX: 0,
    positionAbsoluteY: 0,
  } as NodeProps;

  render(
    <ReactFlowProvider>
      <TapNode {...props} />
    </ReactFlowProvider>,
  );
}

describe('TapNode', () => {
  it('renders a selected probe tap with output handles', () => {
    renderTapNode(
      {
        tap: {
          id: 'probe-1',
          type: 'probe',
          position: { afterNode: 'cell' },
          paths: {
            hidden: 'state.hidden',
            output: 'ports.output',
          },
        },
      },
      true,
    );

    expect(screen.getByTitle('probe')).toHaveTextContent('P');
    expect(document.querySelectorAll('.react-flow__handle')).toHaveLength(2);
  });
});
