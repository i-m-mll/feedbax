import { Activity, X } from 'lucide-react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { useState } from 'react';

import { stopTraining } from '@/api/client';

interface Job {
  id: string;
  family: 'training' | 'analysis';
  status: string;
  error?: string | null;
  manifest_id?: string | null;
  completed_batches?: number | null;
  total_batches?: number | null;
  event_sequence?: number;
  resumable?: boolean;
}

async function jobs(): Promise<Job[]> {
  const response = await fetch('/api/desktop/jobs');
  if (!response.ok) throw new Error('Studio jobs are unavailable. Select Reconnect in the app menu.');
  const data = await response.json() as {
    training: (Omit<Job, 'family' | 'id'> & { job_id: string })[];
    analysis: (Omit<Job, 'family' | 'id'> & { request_id: string })[];
  };
  return [
    ...data.training.map((job) => ({ ...job, family: 'training' as const, id: job.job_id })),
    ...data.analysis.map((job) => ({ ...job, family: 'analysis' as const, id: job.request_id })),
  ];
}

export function DesktopJobs() {
  const [open, setOpen] = useState(false);
  const list = useQuery({
    queryKey: ['desktop-jobs'], queryFn: jobs, enabled: open, refetchInterval: open ? 1500 : false,
  });
  const action = useMutation({
    mutationFn: async ({ job, resume }: { job: Job; resume: boolean }) => {
      if (job.family === 'training' && !resume) {
        await stopTraining(job.id);
        return;
      }
      const id = encodeURIComponent(job.id);
      const response = await fetch(
        resume ? `/api/desktop/jobs/${id}/resume` : `/api/analyses/jobs/${id}`,
        { method: resume ? 'POST' : 'DELETE' },
      );
      if (!response.ok) {
        const body = await response.json() as { detail?: string };
        throw new Error(body.detail ?? 'Studio could not change this job.');
      }
    },
    onSuccess: () => void list.refetch(),
  });
  return (
    <div className="relative">
      <button
        type="button"
        title="Background jobs"
        aria-label="Background jobs"
        aria-expanded={open}
        className="p-1.5 rounded-full hover:bg-slate-100"
        onClick={() => setOpen(!open)}
      >
        <Activity className="w-4 h-4" />
      </button>
      {open && (
        <section aria-label="Background jobs" className="absolute z-50 right-0 top-9 w-96 max-h-96 overflow-auto rounded-md border bg-white p-3 shadow-lg text-sm text-slate-900">
          <div className="flex items-center justify-between">
            <h2 className="font-medium">Background jobs</h2>
            <button type="button" aria-label="Close background jobs" onClick={() => setOpen(false)}>
              <X className="w-4 h-4" />
            </button>
          </div>
          {list.isPending && <p>Loading jobs…</p>}
          {list.error && <p role="alert">{list.error.message}</p>}
          {action.error && <p role="alert">{action.error.message}</p>}
          {list.data?.length === 0 && <p>No background jobs yet.</p>}
          <ul>
            {list.data?.map((job) => {
              const active = ['claimed', 'pending', 'launched', 'ready', 'running'].includes(job.status);
              return (
                <li key={`${job.family}:${job.id}`} className="mt-2 border-t pt-2">
                  <p className="capitalize">{job.family} · {job.status.replace(/_/g, ' ')}</p>
                  <p className="break-all text-xs text-slate-600">{job.manifest_id ?? job.id}</p>
                  {job.completed_batches != null && (
                    <p>{job.completed_batches}{job.total_batches ? ` of ${job.total_batches}` : ''} batches</p>
                  )}
                  {active && job.event_sequence != null && job.event_sequence >= 0 && (
                    <p>Last event {job.event_sequence}</p>
                  )}
                  {job.error && <p className="break-words text-red-700">{job.error}</p>}
                  {(active || job.resumable) && (
                    <button
                      type="button"
                      className="mt-1 rounded border px-2 py-1"
                      disabled={action.isPending}
                      onClick={() => action.mutate({ job, resume: !active })}
                    >
                      {active ? 'Cancel job' : 'Resume from checkpoint'}
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
}
