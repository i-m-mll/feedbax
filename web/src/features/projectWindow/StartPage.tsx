import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { BookOpen, Plus, Search } from 'lucide-react';
import { useId, useState, type ReactNode } from 'react';
import { createProjectFolder, deleteProjectFolder, listProjectExamples, listProjectFolders } from '@/api/projectFolders';
import { shownError } from '@/api/shownErrors';
import {
  studioAccessibleName, studioDateTime, studioFormat, studioName, studioReason, type StudioNameId,
} from '@/components/activity/studioVocabulary';
import { Button } from '@/components/ui/Button';
import { Dialog } from '@/components/ui/Dialog';
import { Menu, MenuContent, MenuItem, MenuTrigger } from '@/components/ui/Menu';
import { Table, TableCell, TableHeaderCell, TableRow, TableRowHeader } from '@/components/ui/Table';
import { TextField } from '@/components/ui/TextField';
import {
  StudioProjectNewSchema, type StudioProjectExample, type StudioProjectListEntry,
} from '@/generated/studioContracts';
import { ProjectlessWindow } from './ProjectWindow';

const projectsKey = ['studio-home-projects'];

function projectHref(id: string) {
  return `?project=${encodeURIComponent(id)}`;
}

/** An action that leaves the start page for a project's window: it runs `invoke`, or opens the menu of `items`. */
type StartAction = { name: StudioNameId; glyph: ReactNode; disabled?: boolean } & (
  { invoke(): void } | { items: ReactNode }
);

function ActionButton({ action }: { action: StartAction }) {
  const button = <Button variant="ghost" align="start" className="w-full gap-2" disabled={action.disabled}
    onClick={'invoke' in action ? action.invoke : undefined}>{action.glyph}{studioName(action.name)}</Button>;
  if ('invoke' in action) return button;
  return <Menu>
    <MenuTrigger render={button} />
    <MenuContent aria-label={studioName(action.name)} side="right">{action.items}</MenuContent>
  </Menu>;
}

/** A new project's name, and the example it copies. */
interface Naming {
  name: string;
  example?: StudioProjectExample;
}

function NameDialog({ naming, onChange, onClose }: { naming: Naming; onChange(naming: Naming): void; onClose(): void }) {
  const [invalid, setInvalid] = useState(false);
  const problemId = useId();
  const create = useMutation({
    mutationFn: ({ name, example }: Naming) => createProjectFolder({ name, example: example?.id }),
    onSuccess: (document) => window.location.assign(projectHref(document.snapshot.project_id)),
  });
  const busy = create.isPending || create.isSuccess;
  const label = studioName('newProject');
  const problem = invalid ? studioReason('projectNameCharacters', studioName('project'))
    : create.isError ? shownError(create.error) : undefined;
  return <Dialog ariaLabel={label} manageFocus onClose={() => { if (!busy) onClose(); }}
    panelClassName="w-[min(22rem,90vw)]">
    <form className="grid gap-2" onSubmit={(event) => {
      event.preventDefault();
      if (!StudioProjectNewSchema.safeParse({ name: naming.name }).success) {
        setInvalid(true);
        return;
      }
      create.mutate(naming);
    }}>
      <label className="grid gap-1">
        {studioName('name')}
        <TextField value={naming.name} required data-dialog-autofocus disabled={busy}
          invalid={!!problem} aria-describedby={problem ? problemId : undefined}
          onValueChange={(name) => {
            onChange({ ...naming, name });
            setInvalid(false);
            create.reset();
          }} />
      </label>
      {problem && <p id={problemId} role="alert">{problem}</p>}
      <Button type="submit" variant="primary" className="justify-self-end" disabled={busy}>{label}</Button>
    </form>
  </Dialog>;
}

function DeleteDialog({ project, onClose }: { project: StudioProjectListEntry; onClose(): void }) {
  const queries = useQueryClient();
  const remove = useMutation({
    mutationFn: () => deleteProjectFolder(project.project_id),
    onSuccess: async () => {
      await queries.invalidateQueries({ queryKey: projectsKey });
      onClose();
    },
  });
  const title = studioFormat('deleteProject', { name: project.name });
  return <Dialog ariaLabel={title} manageFocus onClose={() => { if (!remove.isPending) onClose(); }}
    panelClassName="grid w-80 gap-3">
    <h2 className="font-medium">{title}</h2>
    <p className="text-shell-muted"><FolderPath folder={project.folder} /></p>
    {remove.isError && <p role="alert">{shownError(remove.error)}</p>}
    <div className="flex justify-end gap-2">
      <Button onClick={onClose} disabled={remove.isPending}>{studioName('cancel')}</Button>
      <Button variant="primary" data-dialog-autofocus disabled={remove.isPending}
        onClick={() => remove.mutate()}>{studioName('delete')}</Button>
    </div>
  </Dialog>;
}

/** A folder's path, cut at its start when it is too long, so that its own name stays in view. */
function FolderPath({ folder }: { folder: string }) {
  return <span dir="rtl" className="block truncate text-left"><bdi dir="ltr">{folder}</bdi></span>;
}

function matches(project: StudioProjectListEntry, filter: string) {
  const term = filter.trim().toLocaleLowerCase();
  return !term || project.name.toLocaleLowerCase().includes(term) || project.folder.toLocaleLowerCase().includes(term);
}

/**
 * The window with no project, under a modal that cannot be dismissed: its actions and its recent projects, the most
 * recently modified first. Opening a project, or making one, is the only way out.
 */
export function StartPage() {
  const projects = useQuery({ queryKey: projectsKey, queryFn: ({ signal }) => listProjectFolders(signal) });
  const examples = useQuery({ queryKey: ['studio-project-examples'], queryFn: ({ signal }) => listProjectExamples(signal) });
  const [filter, setFilter] = useState('');
  const [naming, setNaming] = useState<Naming>();
  const [deleting, setDeleting] = useState<StudioProjectListEntry>();
  const rows = [...projects.data ?? []].filter((project) => matches(project, filter))
    .sort((a, b) => Date.parse(b.modified) - Date.parse(a.modified));
  const actions: StartAction[] = [
    { name: 'newProject', glyph: <Plus aria-hidden="true" />, invoke: () => setNaming({ name: '' }) },
    {
      name: 'examples', glyph: <BookOpen aria-hidden="true" />, disabled: !examples.data && !examples.isError,
      items: examples.isError
        ? <MenuItem disabled>{shownError(examples.error)}</MenuItem>
        : examples.data?.map((example) => (
          <MenuItem key={example.id} onClick={() => setNaming({ name: example.id, example })}>{example.name}</MenuItem>
        )),
    },
  ];
  return <>
    <ProjectlessWindow pending={false} />
    <Dialog ariaLabel={studioName('studio')} manageFocus closeOnBackdrop={false} closeOnEscape={false} onClose={() => {}}
      panelClassName="grid h-[min(36rem,88vh)] w-[min(60rem,94vw)] grid-cols-[11rem_minmax(0,1fr)] gap-3">
      <nav aria-label={studioName('studio')} className="grid content-start gap-1">
        {actions.map((action) => <ActionButton key={action.name} action={action} />)}
      </nav>
      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)] gap-2">
        <TextField aria-label={studioName('filter')} placeholder={studioName('filter')} leading={<Search />}
          value={filter} onValueChange={setFilter} data-dialog-autofocus />
        {projects.isError ? <div className="grid content-start justify-items-start gap-2">
          <p role="alert">{shownError(projects.error)}</p>
          <Button onClick={() => void projects.refetch()}>{studioName('retry')}</Button>
        </div> : (
          <Table aria-label={studioName('projects')} held={1} aria-busy={projects.isPending || undefined}
            columns={[{ least: '10rem' }, { least: '9rem' }, { least: '4rem' }, { least: '8rem', greatest: '20rem' }]}>
            <thead><tr>
              <TableHeaderCell scope="col">{studioName('name')}</TableHeaderCell>
              <TableHeaderCell scope="col">{studioName('modified')}</TableHeaderCell>
              <TableHeaderCell scope="col" className="text-right">{studioName('steps')}</TableHeaderCell>
              <TableHeaderCell scope="col">{studioName('folder')}</TableHeaderCell>
            </tr></thead>
            <tbody>{rows.map((project) => (
              <TableRow key={project.project_id} menu={{
                label: studioAccessibleName('project', project.name),
                items: <MenuItem onClick={() => setDeleting(project)}>{studioName('delete')}</MenuItem>,
              }}>
                <TableRowHeader><a href={projectHref(project.project_id)} className="block truncate">{project.name}</a></TableRowHeader>
                <TableCell className="tabular-nums text-shell-muted">{studioDateTime(Date.parse(project.modified))}</TableCell>
                <TableCell className="text-right tabular-nums text-shell-muted">{project.step_count}</TableCell>
                <TableCell className="text-shell-muted"><FolderPath folder={project.folder} /></TableCell>
              </TableRow>
            ))}</tbody>
          </Table>
        )}
      </div>
      {naming && <NameDialog naming={naming} onChange={setNaming} onClose={() => setNaming(undefined)} />}
      {deleting && <DeleteDialog project={deleting} onClose={() => setDeleting(undefined)} />}
    </Dialog>
  </>;
}
