import { useTrainingStore } from '@/stores/trainingStore';
import { useTraining } from '@/hooks/useTraining';
import { useMemo } from 'react';
import { useGraphStore } from '@/stores/graphStore';

export function TrainingPanel() {
  const { trainingSpec, setTrainingSpec, progress, status } = useTrainingStore();
  const { start, stop } = useTraining();
  const graphId = useGraphStore((state) => state.graphId);

  const percent = useMemo(() => {
    if (!progress) return 0;
    return Math.round((progress.batch / progress.total_batches) * 100);
  }, [progress]);

  return (
    <div className="p-6 space-y-4 text-sm text-slate-600 overflow-x-hidden">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Training</div>
        <div className="text-base font-semibold text-slate-800">Configuration</div>
      </div>
      <div className="space-y-3">
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-2">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Optimizer</div>
          <div className="flex items-center gap-2">
            <select
              value={trainingSpec.optimizer.type}
              onChange={(event) =>
                setTrainingSpec({
                  optimizer: { ...trainingSpec.optimizer, type: event.target.value },
                })
              }
              className="rounded-lg border border-slate-200 px-2 py-1 text-sm"
            >
              <option value="adam">Adam</option>
              <option value="sgd">SGD</option>
              <option value="adamw">AdamW</option>
            </select>
            <input
              type="number"
              step="0.0001"
              value={Number(trainingSpec.optimizer.params.learning_rate ?? 0.001)}
              onChange={(event) =>
                setTrainingSpec({
                  optimizer: {
                    ...trainingSpec.optimizer,
                    params: {
                      ...trainingSpec.optimizer.params,
                      learning_rate: Number(event.target.value),
                    },
                  },
                })
              }
              className="w-24 rounded-lg border border-slate-200 px-2 py-1 text-sm"
            />
            <span className="text-xs text-slate-400">lr</span>
          </div>
        </div>
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Loss</div>
          <div className="text-sm text-slate-600">Composite loss tree</div>
        </div>
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-2">
          <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Batches</div>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min={1}
              value={trainingSpec.n_batches}
              onChange={(event) => setTrainingSpec({ n_batches: Number(event.target.value) })}
              className="w-24 rounded-lg border border-slate-200 px-2 py-1 text-sm"
            />
            <span className="text-xs text-slate-400">batches</span>
            <input
              type="number"
              min={1}
              value={trainingSpec.batch_size}
              onChange={(event) => setTrainingSpec({ batch_size: Number(event.target.value) })}
              className="w-20 rounded-lg border border-slate-200 px-2 py-1 text-sm"
            />
            <span className="text-xs text-slate-400">batch size</span>
          </div>
        </div>
      </div>

      {progress && (
        <div className="rounded-xl border border-slate-100 bg-white p-4 space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>Progress</span>
            <span>
              {progress.batch}/{progress.total_batches}
            </span>
          </div>
          <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
            <div
              className="h-full bg-brand-500 transition-all"
              style={{ width: `${percent}%` }}
            />
          </div>
          <div className="text-xs text-slate-500">Loss: {progress.loss.toFixed(4)}</div>
        </div>
      )}
      {status === 'completed' && (
        <div className="text-xs text-mint-500">Training completed.</div>
      )}
      {status === 'error' && (
        <div className="text-xs text-amber-600">Training failed. Check console.</div>
      )}

      {!graphId && (
        <div className="text-xs text-amber-600">Save the project before starting training.</div>
      )}
      <button
        className="w-full rounded-full bg-brand-500 text-white py-2 text-sm font-semibold shadow-soft hover:bg-brand-600"
        onClick={status === 'running' ? stop : start}
        disabled={!graphId}
      >
        {status === 'running' ? 'Stop Training' : 'Start Training'}
      </button>
    </div>
  );
}
