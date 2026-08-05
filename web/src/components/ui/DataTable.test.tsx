// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import {
  DataTable,
  DataTableCell,
  DataTableRow,
} from './DataTable';
import { ValueCountBadge } from './ValueCountBadge';

describe('DataTable', () => {
  it('uses one semantic column contract inside a horizontal overflow container', () => {
    render(
      <DataTable
        aria-label="Example values"
        columns={[
          { id: 'name', label: 'Name', width: '12rem' },
          { id: 'value', label: 'Value', width: '8rem' },
        ]}
        minWidth="20rem"
      >
        <DataTableRow>
          <DataTableCell>Learning rate</DataTableCell>
          <DataTableCell>0.001</DataTableCell>
        </DataTableRow>
      </DataTable>
    );

    const table = screen.getByRole('table', { name: 'Example values' });
    expect(table).toHaveStyle({ minWidth: '20rem' });
    expect(screen.getAllByRole('columnheader').map((header) => header.textContent)).toEqual([
      'Name',
      'Value',
    ]);
    expect(screen.getByTestId('data-table-scroll-container')).toHaveClass('overflow-x-auto');
    expect(table.querySelectorAll('col')).toHaveLength(2);
  });
});

describe('ValueCountBadge', () => {
  it('shows only the compact number while exposing the full value-count label', () => {
    render(<ValueCountBadge count={2} />);

    const badge = screen.getByLabelText('2 values');
    expect(badge).toHaveTextContent('2');
    expect(badge).not.toHaveTextContent('values');
  });
});
