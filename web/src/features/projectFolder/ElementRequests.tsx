import { studioName, studioRunReason, studioRunStatus } from '@/components/activity/studioVocabulary';
import { StatusMark } from '@/components/ui/StatusMark';
import { Button } from '@/components/ui/Button';
import { Table, TableCell, TableHeaderCell, TableRow, TableRowHeader } from '@/components/ui/Table';
import type { StudioProjectWindowMapRequest } from '@/generated/studioContracts';

/** A request's member id, which tells it apart among its element's requests. */
export function requestMember(request: StudioProjectWindowMapRequest) {
  return JSON.stringify(Object.entries(request.labels ?? {}));
}

/** The requests of an element that maps over labels, one row each by its label values, as a step's runs are listed. */
export function ElementRequests({ requests, selected, onSelect }: {
  requests: readonly StudioProjectWindowMapRequest[];
  selected?: string;
  onSelect(member: string | undefined): void;
}) {
  const labels = [...new Set(requests.flatMap((request) => Object.keys(request.labels ?? {})))];
  const mark = { least: '1.75rem', greatest: '1.75rem' };
  return <Table aria-label={studioName('runs')} held={1}
    columns={[mark, ...labels.map(() => ({ least: '6rem' })), mark]}>
    <thead><tr>
      <TableHeaderCell />
      {labels.map((label) => <TableHeaderCell key={label}>{label}</TableHeaderCell>)}
      <TableHeaderCell />
    </tr></thead>
    <tbody>{requests.map((request, index) => {
      const member = requestMember(request);
      const lit = member === selected;
      const values = labels.map((label) => request.labels?.[label] ?? '');
      return <TableRow key={member} selected={lit}>
        <TableRowHeader className="pl-1 pr-2 text-right">
          <Button variant="bare" size="content" pressed={lit}
            className="w-full justify-end rounded-row tabular-nums text-shell-muted"
            onClick={() => onSelect(lit ? undefined : member)}>
            {index + 1}<span className="sr-only"> {values.join(' ')}</span>
          </Button>
        </TableRowHeader>
        {values.map((value, column) => <TableCell key={labels[column]}>{value}</TableCell>)}
        <TableCell><StatusMark status={request.status} label={studioRunStatus(request.status)}
          tip={request.reason ? studioRunReason({ reason: request.reason }) : undefined} /></TableCell>
      </TableRow>;
    })}</tbody>
  </Table>;
}
