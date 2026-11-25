import { useMutation, useQuery } from '@tanstack/react-query';
import { useId, useState } from 'react';
import { createProjectFolder, listProjectFolders } from '@/api/projectFolders';
import { shownError } from '@/api/shownErrors';
import { studioDateTime, studioName, studioReason } from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Table, TableCell, TableHeaderCell, TableRow, TableRowHeader } from '@/components/ui/Table';
import { TextField } from '@/components/ui/TextField';
import { StudioProjectNewSchema } from '@/generated/studioContracts';

function projectHref(id: string) {
  return `?project=${encodeURIComponent(id)}`;
}

/** The Studio home's projects, the most recently modified first, each opening its window; `+` makes a new one. */
export function ProjectHome() {
  const projects = useQuery({ queryKey: ['studio-home-projects'], queryFn: ({ signal }) => listProjectFolders(signal) });
  const [name, setName] = useState<string>();
  const [invalid, setInvalid] = useState(false);
  const problemId = useId();
  const create = useMutation({
    mutationFn: (name: string) => createProjectFolder({ name }),
    onSuccess: (document) => window.location.assign(projectHref(document.snapshot.project_id)),
  });
  const rows = [...projects.data ?? []].sort((a, b) => Date.parse(b.modified) - Date.parse(a.modified));
  const problem = invalid ? studioReason('projectNameCharacters', studioName('project'))
    : create.isError ? shownError(create.error) : undefined;
  return (
    <main className="grid h-screen content-start gap-2 overflow-auto bg-shell-surface p-3 text-shell-foreground">
      <h1 className="font-medium">{studioName('projects')}</h1>
      {projects.isError ? <>
        <p role="alert">{shownError(projects.error)}</p>
        <Button className="justify-self-start" onClick={() => void projects.refetch()}>{studioName('retry')}</Button>
      </> : (
        <Table aria-label={studioName('projects')} held={1} columns={[{ least: '12rem' }, { least: '10rem' }]}
          addRow={{ label: studioName('newProject'), onAdd: () => {
            create.reset();
            setInvalid(false);
            setName('');
          }, disabled: create.isPending }}
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
      {name !== undefined && <Dialog ariaLabel={studioName('newProject')} manageFocus
        onClose={() => { if (!create.isPending) setName(undefined); }}
        panelClassName="w-[min(22rem,90vw)]">
        <form className="grid gap-2" onSubmit={(event) => {
          event.preventDefault();
          if (!StudioProjectNewSchema.safeParse({ name }).success) {
            setInvalid(true);
            return;
          }
          create.mutate(name);
        }}>
          <label className="grid gap-1">
            {studioName('name')}
            <TextField value={name} required data-dialog-autofocus disabled={create.isPending}
              invalid={!!problem} aria-describedby={problem ? problemId : undefined}
              onValueChange={(value) => {
                setName(value);
                setInvalid(false);
                create.reset();
              }} />
          </label>
          {problem && <p id={problemId} role="alert">{problem}</p>}
          <Button type="submit" className="justify-self-end" disabled={create.isPending}>
            {studioName('newProject')}
          </Button>
        </form>
      </Dialog>}
    </main>
  );
}
