import { useMutation, useQuery } from '@tanstack/react-query';
import { createProjectFolder, listProjectFolders } from '@/api/projectFolders';
import { shownError } from '@/api/shownErrors';
import { studioDateTime, studioName } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Table, TableCell, TableHeaderCell, TableRow, TableRowHeader } from '@/components/ui/Table';

function projectHref(id: string) {
  return `?project=${encodeURIComponent(id)}`;
}

/** The Studio home's projects, the most recently modified first, each opening its window; `+` makes a new one. */
export function ProjectHome() {
  const projects = useQuery({ queryKey: ['studio-home-projects'], queryFn: ({ signal }) => listProjectFolders(signal) });
  const create = useMutation({
    mutationFn: () => createProjectFolder(),
    onSuccess: (document) => window.location.assign(projectHref(document.snapshot.project_id)),
  });
  const rows = [...projects.data ?? []].sort((a, b) => Date.parse(b.modified) - Date.parse(a.modified));
  return (
    <main className="grid h-screen content-start gap-2 overflow-auto bg-shell-surface p-3 text-shell-foreground">
      <h1 className="font-medium">{studioName('projects')}</h1>
      {projects.isError ? <>
        <p role="alert">{shownError(projects.error)}</p>
        <Button className="justify-self-start" onClick={() => void projects.refetch()}>{studioName('retry')}</Button>
      </> : (
        <Table aria-label={studioName('projects')} held={1} columns={[{ least: '12rem' }, { least: '10rem' }]}
          addRow={{ label: studioName('newProject'), onAdd: () => create.mutate(), disabled: create.isPending }}
          aria-busy={projects.isPending || undefined}>
          <thead><tr>
            <TableHeaderCell scope="col">{studioName('name')}</TableHeaderCell>
            <TableHeaderCell scope="col">{studioName('modified')}</TableHeaderCell>
          </tr></thead>
          <tbody>{rows.map(({ project_id, name, modified }) => (
            <TableRow key={project_id}>
              <TableRowHeader><a href={projectHref(project_id)} className="block truncate">{name}</a></TableRowHeader>
              <TableCell className="tabular-nums text-shell-muted">{studioDateTime(Date.parse(modified))}</TableCell>
            </TableRow>
          ))}</tbody>
        </Table>
      )}
      {create.isError && <p role="alert">{shownError(create.error)}</p>}
    </main>
  );
}
