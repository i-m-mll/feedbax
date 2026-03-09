import { useTrainingStore } from '@/stores/trainingStore';
import { TrajectoryViewer } from './TrajectoryViewer';
import { useTraining, extractNetworkParams } from '@/hooks/useTraining';
import { useWorkerConfig } from '@/hooks/useWorkerConfig';
import { useOrchestration } from '@/hooks/useOrchestration';
import { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import type { LossTermSpec, TimeAggregationSpec } from '@/types/training';
import { LossTermDetail } from './LossTermDetail';
import { AddLossTermModal } from '@/components/modals/AddLossTermModal';
import { fetchProbes, validateLossSpec, downloadCheckpoint } from '@/api/client';
import clsx from 'clsx';
import { Plus, Trash2, AlertCircle, ChevronDown, ChevronRight, Download, Loader2 } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const LOSS_TERM_COLORS = ['#f59e0b', '#10b981', '#ef4444', '#8b5cf6'];

export function TrainingPanel() {
  const { trainingSpec, setTrainingSpec, progress, status, lossHistory, jobId, latestTrajectory } = useTrainingStore();
  const setAvailableProbes = useTrainingStore((state) => state.setAvailableProbes);
  const setLossValidationErrors = useTrainingStore((state) => state.setLossValidationErrors);
  const lossValidationErrors = useTrainingStore((state) => state.lossValidationErrors);
  const removeLossTerm = useTrainingStore((state) => state.removeLossTerm);
  const setHighlightedProbeSelector = useTrainingStore((state) => state.setHighlightedProbeSelector);
  const { start, stop } = useTraining();
  const { workerMode, workerUrl, workerConnected, connecting, error: workerError, connect } =
    useWorkerConfig();
  const {
    status: cloudStatus,
    instanceName: cloudInstanceName,
    workerUrl: cloudWorkerUrl,
    launching: cloudLaunching,
    terminating: cloudTerminating,
    error: cloudError,
    launch: cloudLaunch,
    terminate: cloudTerminate,
  } = useOrchestration();
  const graphId = useGraphStore((state) => state.graphId);
  const graph = useGraphStore((state) => state.graph);
  const inSubgraph = useGraphStore((state) => state.graphStack.length > 0);

  // Derived network params for the config summary chip row
  const networkParams = useMemo(() => extractNetworkParams(graph), [graph]);
  const [expanded, setExpanded] = useState<Record<string, boolean>>(() =>
    buildDefaultExpanded(trainingSpec.loss)
  );
  const [selectedPath, setSelectedPath] = useState<string | null>(null);
  const [showAddModal, setShowAddModal] = useState(false);
  const [addModalParentPath, setAddModalParentPath] = useState<string[]>([]);
  const [showDetailPanel, setShowDetailPanel] = useState(false);
  const nodeRefs = useRef<Record<string, HTMLDivElement | null>>({});
  // Remote worker UI state
  const [remoteExpanded, setRemoteExpanded] = useState(false);
  const [remoteUrl, setRemoteUrl] = useState('');
  const [remoteToken, setRemoteToken] = useState('');
  // Cloud panel UI state
  const [cloudExpanded, setCloudExpanded] = useState(false);
  const [cloudProject, setCloudProject] = useState('');
  const [cloudZone, setCloudZone] = useState('us-central1-a');
  const [cloudMachineType, setCloudMachineType] = useState('n1-standard-4');
  const [cloudPreemptible, setCloudPreemptible] = useState(true);
  const [cloudWorkerPort, setCloudWorkerPort] = useState(8765);
  const [cloudAuthToken, setCloudAuthToken] = useState('');
  const [cloudTsAuthKey, setCloudTsAuthKey] = useState('');

  // Fetch available probes when graph changes
  useEffect(() => {
    if (graphId) {
      fetchProbes(graphId)
        .then(setAvailableProbes)
        .catch((err) => console.warn('Failed to fetch probes:', err));
    }
  }, [graphId, setAvailableProbes]);

  // Validate loss spec when it changes
  useEffect(() => {
    if (graphId) {
      validateLossSpec(graphId, trainingSpec.loss)
        .then((result) => setLossValidationErrors(result.errors))
        .catch((err) => console.warn('Failed to validate loss:', err));
    }
  }, [graphId, trainingSpec.loss, setLossValidationErrors]);

  const percent = useMemo(() => {
    if (!progress) return 0;
    return Math.round((progress.batch / progress.total_batches) * 100);
  }, [progress]);

  const equationTerms = useMemo(
    () => collectEquationTerms(trainingSpec.loss),
    [trainingSpec.loss]
  );

  useEffect(() => {
    setExpanded((prev) => ({
      ...buildDefaultExpanded(trainingSpec.loss),
      ...prev,
    }));
  }, [trainingSpec.loss]);

  const registerNode = useCallback((pathKey: string, node: HTMLDivElement | null) => {
    nodeRefs.current[pathKey] = node;
  }, []);

  const handleToggle = useCallback((path: string[]) => {
    const key = toPathKey(path);
    setExpanded((prev) => ({
      ...prev,
      [key]: !(prev[key] ?? true),
    }));
  }, []);

  const handleSelect = useCallback((path: string[]) => {
    setSelectedPath(toPathKey(path));
  }, []);

  const handleWeightChange = useCallback(
    (path: string[], weight: number) => {
      if (!Number.isFinite(weight)) return;
      const updated = updateLossWeight(trainingSpec.loss, path, weight);
      setTrainingSpec({ loss: updated });
    },
    [setTrainingSpec, trainingSpec.loss]
  );

  const handleJumpToTerm = useCallback((path: string[]) => {
    const key = toPathKey(path);
    setSelectedPath(key);
    setExpanded((prev) => {
      const next = { ...prev, [ROOT_PATH]: true };
      for (let i = 1; i <= path.length; i += 1) {
        next[path.slice(0, i).join('/')] = true;
      }
      return next;
    });
    nodeRefs.current[key]?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }, []);

  const handleAddTerm = useCallback((parentPath: string[]) => {
    setAddModalParentPath(parentPath);
    setShowAddModal(true);
  }, []);

  const handleRemoveTerm = useCallback(
    (path: string[]) => {
      if (path.length === 0) return; // Cannot remove root
      if (confirm('Remove this loss term?')) {
        removeLossTerm(path);
        if (selectedPath === toPathKey(path)) {
          setSelectedPath(null);
          setShowDetailPanel(false);
        }
      }
    },
    [removeLossTerm, selectedPath]
  );

  const handleOpenDetail = useCallback((path: string[]) => {
    setSelectedPath(toPathKey(path));
    setShowDetailPanel(true);
  }, []);

  const handleCloseDetail = useCallback(() => {
    setShowDetailPanel(false);
  }, []);

  const handleTermHover = useCallback(
    (term: LossTermSpec | null) => {
      if (term?.selector) {
        setHighlightedProbeSelector(term.selector);
      } else {
        setHighlightedProbeSelector(null);
      }
    },
    [setHighlightedProbeSelector]
  );

  const selectedPathArray = useMemo(() => {
    if (!selectedPath || selectedPath === ROOT_PATH) return [];
    return selectedPath.split('/');
  }, [selectedPath]);

  return (
    <div className="p-6 space-y-4 text-sm text-slate-600 overflow-x-hidden">
      <div>
        <div className="text-xs uppercase tracking-[0.3em] text-slate-400">Training</div>
        <div className="text-base font-semibold text-slate-800">Configuration</div>
      </div>
      <div className="space-y-3">
        {/* Remote Worker section */}
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 overflow-hidden">
          <button
            type="button"
            className="flex w-full items-center justify-between px-4 py-3 text-left"
            onClick={() => setRemoteExpanded((v) => !v)}
          >
            <div className="flex items-center gap-2">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Remote Worker</div>
              {/* Status chip */}
              <span
                className={clsx(
                  'rounded-full px-2 py-0.5 text-[10px] font-medium',
                  workerMode === 'remote' && workerConnected
                    ? 'bg-emerald-100 text-emerald-700'
                    : workerMode === 'remote'
                    ? 'bg-amber-100 text-amber-700'
                    : 'bg-slate-100 text-slate-500'
                )}
              >
                {workerMode === 'remote' ? (workerConnected ? 'remote connected' : 'remote disconnected') : 'local'}
              </span>
            </div>
            {remoteExpanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
            )}
          </button>
          {remoteExpanded && (
            <div className="border-t border-slate-100 px-4 pb-4 pt-3 space-y-2">
              <input
                type="url"
                placeholder="http://100.x.x.x:8765"
                value={remoteUrl}
                onChange={(e) => setRemoteUrl(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm placeholder-slate-300"
              />
              <input
                type="password"
                placeholder="Auth token (optional)"
                value={remoteToken}
                onChange={(e) => setRemoteToken(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm placeholder-slate-300"
              />
              {workerError && (
                <div className="text-xs text-red-500">{workerError}</div>
              )}
              <button
                type="button"
                disabled={connecting || !remoteUrl.trim()}
                onClick={() => connect(remoteUrl.trim(), remoteToken.trim() || null)}
                className="w-full rounded-lg bg-brand-500 py-1.5 text-xs font-semibold text-white disabled:opacity-50 hover:bg-brand-600 transition-colors"
              >
                {connecting ? 'Connecting…' : 'Connect'}
              </button>
              {workerUrl && (
                <div className="truncate text-[10px] text-slate-400" title={workerUrl}>
                  {workerUrl}
                </div>
              )}
            </div>
          )}
        </div>
        {/* Cloud panel */}
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 overflow-hidden">
          <button
            type="button"
            className="flex w-full items-center justify-between px-4 py-3 text-left"
            onClick={() => setCloudExpanded((v) => !v)}
          >
            <div className="flex items-center gap-2">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Cloud</div>
              <CloudStatusChip status={cloudStatus} />
            </div>
            {cloudExpanded ? (
              <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
            ) : (
              <ChevronRight className="w-3.5 h-3.5 text-slate-400" />
            )}
          </button>
          {cloudExpanded && (
            <div className="border-t border-slate-100 px-4 pb-4 pt-3 space-y-2">
              <input
                type="text"
                placeholder="GCP Project (e.g. my-project)"
                value={cloudProject}
                onChange={(e) => setCloudProject(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm placeholder-slate-300"
              />
              <input
                type="text"
                placeholder="Zone (e.g. us-central1-a)"
                value={cloudZone}
                onChange={(e) => setCloudZone(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm placeholder-slate-300"
              />
              <div className="space-y-0.5">
                <input
                  type="text"
                  placeholder="Machine type (e.g. n1-standard-4)"
                  value={cloudMachineType}
                  onChange={(e) => setCloudMachineType(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm placeholder-slate-300"
                />
                <div className="text-[10px] text-slate-400 pl-1">
                  Use <code className="font-mono">ct5lp-hightpu-4t</code> for TPU
                </div>
              </div>
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={cloudPreemptible}
                  onChange={(e) => setCloudPreemptible(e.target.checked)}
                  className="rounded border-slate-300"
                />
                <span className="text-sm text-slate-600">Preemptible</span>
              </label>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min={1}
                  max={65535}
                  value={cloudWorkerPort}
                  onChange={(e) => setCloudWorkerPort(Number(e.target.value))}
                  className="w-24 rounded-lg border border-slate-200 px-2 py-1.5 text-sm"
                />
                <span className="text-xs text-slate-400">worker port</span>
              </div>
              <input
                type="password"
                placeholder="Auth token (optional)"
                value={cloudAuthToken}
                onChange={(e) => setCloudAuthToken(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm placeholder-slate-300"
              />
              <div className="space-y-0.5">
                <input
                  type="password"
                  placeholder="Tailscale auth key (optional)"
                  value={cloudTsAuthKey}
                  onChange={(e) => setCloudTsAuthKey(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 px-2 py-1.5 text-sm placeholder-slate-300"
                />
                <div className="text-[10px] text-slate-400 pl-1">
                  Hint: set <code className="font-mono">FEEDBAX_TS_AUTH_KEY</code> in env
                </div>
              </div>
              {cloudError && (
                <div className="text-xs text-red-500">{cloudError}</div>
              )}
              <div className="flex gap-2">
                <button
                  type="button"
                  disabled={cloudLaunching || cloudStatus === 'creating' || cloudStatus === 'connecting' || !cloudProject.trim()}
                  onClick={() =>
                    cloudLaunch({
                      project: cloudProject.trim(),
                      zone: cloudZone.trim(),
                      machine_type: cloudMachineType.trim(),
                      preemptible: cloudPreemptible,
                      worker_port: cloudWorkerPort,
                      auth_token: cloudAuthToken.trim() || null,
                      ts_auth_key: cloudTsAuthKey.trim() || null,
                    })
                  }
                  className="flex-1 flex items-center justify-center gap-1.5 rounded-lg bg-emerald-500 py-1.5 text-xs font-semibold text-white disabled:opacity-50 hover:bg-emerald-600 transition-colors"
                >
                  {(cloudLaunching || cloudStatus === 'creating' || cloudStatus === 'connecting') && (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  )}
                  Launch instance
                </button>
                <button
                  type="button"
                  disabled={cloudTerminating || cloudStatus === 'idle' || cloudInstanceName === null}
                  onClick={cloudTerminate}
                  className="flex-1 rounded-lg bg-red-500 py-1.5 text-xs font-semibold text-white disabled:opacity-50 hover:bg-red-600 transition-colors"
                >
                  {cloudTerminating ? 'Terminating…' : 'Terminate'}
                </button>
              </div>
              {cloudInstanceName && (
                <div className="truncate text-[10px] text-slate-400" title={cloudInstanceName}>
                  {cloudInstanceName}
                </div>
              )}
              {cloudWorkerUrl && (
                <div className="truncate text-[10px] text-slate-400" title={cloudWorkerUrl}>
                  {cloudWorkerUrl}
                </div>
              )}
            </div>
          )}
        </div>
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
        <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-4 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Loss Function</div>
              {lossValidationErrors.length > 0 && (
                <div className="flex items-center gap-1 text-amber-500" title={`${lossValidationErrors.length} validation error(s)`}>
                  <AlertCircle className="w-3.5 h-3.5" />
                  <span className="text-xs">{lossValidationErrors.length}</span>
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => handleAddTerm([])}
              className="h-7 w-7 rounded-full border border-slate-200 text-slate-500 hover:bg-white hover:text-brand-500 hover:border-brand-200 transition-colors"
              title="Add loss term"
            >
              <Plus className="w-4 h-4 mx-auto" />
            </button>
          </div>
          <LossEquation terms={equationTerms} onSelect={handleJumpToTerm} />
          <LossTree
            term={trainingSpec.loss}
            path={[]}
            depth={0}
            expanded={expanded}
            selectedPath={selectedPath}
            onToggle={handleToggle}
            onSelect={handleSelect}
            onWeightChange={handleWeightChange}
            onAddChild={handleAddTerm}
            onRemove={handleRemoveTerm}
            onOpenDetail={handleOpenDetail}
            onHover={handleTermHover}
            registerNode={registerNode}
          />
        </div>

        {/* Detail panel */}
        {showDetailPanel && selectedPath && (
          <div className="rounded-xl border border-brand-100 bg-white p-4">
            <LossTermDetail
              path={selectedPathArray}
              onClose={handleCloseDetail}
            />
          </div>
        )}

        {/* Add term modal */}
        {showAddModal && (
          <AddLossTermModal
            parentPath={addModalParentPath}
            onClose={() => setShowAddModal(false)}
          />
        )}
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

      <div className="rounded-xl border border-slate-100 bg-white p-4 space-y-3">
        <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Loss</div>
        {lossHistory.length > 0 ? (
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={lossHistory} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis
                dataKey="batch"
                tick={{ fontSize: 10, fill: '#94a3b8' }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                scale="log"
                domain={['auto', 'auto']}
                tick={{ fontSize: 10, fill: '#94a3b8' }}
                tickLine={false}
                axisLine={false}
                width={48}
              />
              <Tooltip
                contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #e2e8f0' }}
                labelFormatter={(v) => `Batch ${v}`}
              />
              <Legend wrapperStyle={{ fontSize: 10 }} />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#6366f1"
                dot={false}
                strokeWidth={1.5}
                name="total loss"
                isAnimationActive={false}
              />
              {lossHistory.length > 0 &&
                Object.keys(lossHistory[0].loss_terms ?? {}).map((termKey, idx) => (
                  <Line
                    key={termKey}
                    type="monotone"
                    dataKey={`loss_terms.${termKey}`}
                    stroke={LOSS_TERM_COLORS[idx % LOSS_TERM_COLORS.length]}
                    dot={false}
                    strokeWidth={1}
                    name={termKey}
                    isAnimationActive={false}
                  />
                ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-[160px] flex items-center justify-center text-xs text-slate-400">
            Waiting for training data...
          </div>
        )}
        {progress && (
          <>
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
          </>
        )}
        {(lossHistory.length > 0 || latestTrajectory !== null) && <TrajectoryViewer />}
      </div>
      {status === 'completed' && (
        <div className="text-xs text-mint-500">Training completed.</div>
      )}
      {status === 'error' && (
        <div className="text-xs text-amber-600">Training failed. Check console.</div>
      )}

      {/* Checkpoint section */}
      <div className="rounded-xl border border-slate-100 bg-white p-4 space-y-2">
        <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Checkpoint</div>
        {progress ? (
          <div className="flex items-center justify-between text-xs text-slate-500">
            <span>
              Batch <span className="font-semibold text-slate-700">{progress.batch}</span>
              {' · '}
              Loss <span className="font-semibold text-slate-700">{progress.loss.toExponential(3)}</span>
            </span>
          </div>
        ) : (
          <div className="text-xs text-slate-400">No checkpoint yet</div>
        )}
        {status === 'completed' && jobId ? (
          <button
            type="button"
            className="flex w-full items-center justify-center gap-2 rounded-lg border border-brand-200 bg-brand-50 py-1.5 text-xs font-semibold text-brand-700 hover:bg-brand-100 transition-colors"
            aria-label="Download checkpoint"
            onClick={async () => {
              try {
                await downloadCheckpoint(jobId);
              } catch (e) {
                console.error('Checkpoint download failed', e);
              }
            }}
          >
            <Download className="w-3.5 h-3.5" />
            Download weights
          </button>
        ) : (
          <div className="relative group">
            <button
              type="button"
              disabled
              className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-200 bg-slate-50 py-1.5 text-xs font-semibold text-slate-400 cursor-not-allowed"
              aria-label="Download checkpoint"
            >
              <Download className="w-3.5 h-3.5" />
              Download checkpoint
            </button>
            <div className="pointer-events-none absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block whitespace-nowrap rounded bg-slate-700 px-2 py-1 text-[10px] text-white shadow">
              Available when connected to real training worker
            </div>
          </div>
        )}
      </div>

      {inSubgraph && (
        <div className="text-xs text-amber-600">
          Return to the model root to start training.
        </div>
      )}
      {!graphId && !inSubgraph && (
        <div className="text-xs text-amber-600">Save the project before starting training.</div>
      )}

      {/* Training config summary — shows what will be sent to the backend */}
      <TrainingConfigSummary
        networkType={networkParams.network_type}
        hiddenDim={networkParams.hidden_dim}
        nBatches={trainingSpec.n_batches}
        batchSize={trainingSpec.batch_size}
        learningRate={
          typeof trainingSpec.optimizer.params.learning_rate === 'number'
            ? trainingSpec.optimizer.params.learning_rate
            : 0.001
        }
      />

      <button
        className="w-full rounded-full bg-brand-500 text-white py-2 text-sm font-semibold shadow-soft hover:bg-brand-600"
        onClick={status === 'running' ? stop : start}
        disabled={!graphId || inSubgraph}
      >
        {status === 'running' ? 'Stop Training' : 'Start Training'}
      </button>
    </div>
  );
}

function LossEquation({
  terms,
  onSelect,
}: {
  terms: EquationTerm[];
  onSelect: (path: string[]) => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs text-slate-600">
      <div className="flex flex-wrap items-center gap-1">
        <span className="font-semibold text-slate-700">L =</span>
        {terms.length === 0 && <span className="text-slate-400">0</span>}
        {terms.map((term, index) => (
          <Fragment key={term.pathKey}>
            {index > 0 && <span className="text-slate-400">+</span>}
            <button
              type="button"
              onClick={() => onSelect(term.path)}
              className="rounded px-1 text-slate-600 hover:bg-slate-100 hover:text-slate-800"
              title={term.label}
            >
              {formatWeight(term.weight)}·L_{term.symbol}
            </button>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function LossTree({
  term,
  path,
  depth = 0,
  expanded,
  selectedPath,
  onToggle,
  onSelect,
  onWeightChange,
  onAddChild,
  onRemove,
  onOpenDetail,
  onHover,
  registerNode,
}: {
  term: LossTermSpec;
  path: string[];
  depth?: number;
  expanded: Record<string, boolean>;
  selectedPath: string | null;
  onToggle: (path: string[]) => void;
  onSelect: (path: string[]) => void;
  onWeightChange: (path: string[], weight: number) => void;
  onAddChild: (parentPath: string[]) => void;
  onRemove: (path: string[]) => void;
  onOpenDetail: (path: string[]) => void;
  onHover: (term: LossTermSpec | null) => void;
  registerNode: (pathKey: string, node: HTMLDivElement | null) => void;
}) {
  const entries = term.children ? Object.entries(term.children) : [];
  const hasChildren = entries.length > 0;
  const pathKey = toPathKey(path);
  const isExpanded = expanded[pathKey] ?? hasChildren;
  const isSelected = selectedPath === pathKey;
  const detailLines = buildDetailLines(term);
  const showDetails = detailLines.length > 0 && !hasChildren;
  const isRoot = path.length === 0;
  const isComposite = term.type === 'Composite' || hasChildren;

  return (
    <div className="space-y-2">
      <div
        ref={(node) => registerNode(pathKey, node)}
        className={clsx(
          'group flex items-center justify-between rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm',
          isSelected && 'border-brand-200 ring-1 ring-brand-200'
        )}
        style={{ marginLeft: depth * 12 }}
        onClick={() => onSelect(path)}
        onDoubleClick={() => onOpenDetail(path)}
        onMouseEnter={() => onHover(term)}
        onMouseLeave={() => onHover(null)}
      >
        <div className="flex items-center gap-2">
          {hasChildren ? (
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onToggle(path);
              }}
              className="text-slate-400 hover:text-slate-600"
              aria-label={isExpanded ? 'Collapse' : 'Expand'}
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          ) : (
            <span className="text-slate-300">•</span>
          )}
          <div className="flex flex-col">
            <span className="font-semibold text-slate-700">{term.label}</span>
            <span className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
              {term.type}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-1 text-xs text-slate-500">
          {/* Action buttons - show on hover */}
          <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity mr-2">
            {isComposite && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onAddChild(path);
                }}
                className="p-1 rounded text-slate-400 hover:text-brand-500 hover:bg-brand-50"
                title="Add child term"
              >
                <Plus className="w-3.5 h-3.5" />
              </button>
            )}
            {!isRoot && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation();
                  onRemove(path);
                }}
                className="p-1 rounded text-slate-400 hover:text-red-500 hover:bg-red-50"
                title="Remove term"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          <span className="text-slate-400">×</span>
          <input
            type="number"
            min={0}
            step={0.001}
            value={Number.isFinite(term.weight) ? term.weight : 0}
            onChange={(event) => onWeightChange(path, Number(event.target.value))}
            onClick={(event) => event.stopPropagation()}
            className="w-20 rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-700"
          />
        </div>
      </div>
      {showDetails && (
        <div
          className="space-y-1 text-xs text-slate-500"
          style={{ marginLeft: depth * 12 + 24 }}
        >
          {detailLines.map((line) => (
            <div key={line}>{line}</div>
          ))}
        </div>
      )}
      {hasChildren && isExpanded && (
        <div className="space-y-2">
          {entries.map(([childLabel, child]) => (
            <LossTree
              key={childLabel}
              term={child}
              path={[...path, childLabel]}
              depth={depth + 1}
              expanded={expanded}
              selectedPath={selectedPath}
              onToggle={onToggle}
              onSelect={onSelect}
              onWeightChange={onWeightChange}
              onAddChild={onAddChild}
              onRemove={onRemove}
              onOpenDetail={onOpenDetail}
              onHover={onHover}
              registerNode={registerNode}
            />
          ))}
        </div>
      )}
    </div>
  );
}

const ROOT_PATH = 'root';

interface EquationTerm {
  path: string[];
  pathKey: string;
  label: string;
  symbol: string;
  weight: number;
}

function toPathKey(path: string[]) {
  return path.length === 0 ? ROOT_PATH : path.join('/');
}

function buildDefaultExpanded(term: LossTermSpec, path: string[] = []): Record<string, boolean> {
  const entries = term.children ? Object.entries(term.children) : [];
  const next: Record<string, boolean> = {
    [toPathKey(path)]: true,
  };
  entries.forEach(([childLabel, child]) => {
    Object.assign(next, buildDefaultExpanded(child, [...path, childLabel]));
  });
  return next;
}

function collectEquationTerms(term: LossTermSpec): EquationTerm[] {
  const terms: EquationTerm[] = [];
  const walk = (node: LossTermSpec, path: string[], weightScale: number) => {
    const nextWeight = weightScale * node.weight;
    const children = node.children ? Object.entries(node.children) : [];
    if (children.length === 0) {
      const key = path[path.length - 1] ?? node.label ?? node.type;
      const label = node.label || key;
      const symbol = sanitizeSymbol(key);
      terms.push({
        path,
        pathKey: toPathKey(path),
        label,
        symbol,
        weight: nextWeight,
      });
      return;
    }
    children.forEach(([childLabel, child]) => {
      walk(child, [...path, childLabel], nextWeight);
    });
  };
  walk(term, [], 1);
  return terms;
}

function updateLossWeight(term: LossTermSpec, path: string[], weight: number): LossTermSpec {
  if (path.length === 0) {
    return { ...term, weight };
  }
  if (!term.children) return term;
  const [head, ...rest] = path;
  const child = term.children[head];
  if (!child) return term;
  return {
    ...term,
    children: {
      ...term.children,
      [head]: updateLossWeight(child, rest, weight),
    },
  };
}

function buildDetailLines(term: LossTermSpec): string[] {
  const selector = formatSelector(term.selector);
  const norm = formatNorm(term.norm);
  const time = formatTimeAggregation(term.time_agg);
  const details: string[] = [];
  const primary = [selector, norm ? `Norm: ${norm}` : null].filter(Boolean).join(' • ');
  if (primary) details.push(primary);
  if (time) details.push(`Time: ${time}`);
  return details;
}

function formatSelector(selector?: string): string | null {
  if (!selector) return null;
  if (selector.startsWith('probe:')) {
    return `Probe: ${selector.slice('probe:'.length)}`;
  }
  return `Path: ${selector}`;
}

function formatNorm(norm?: LossTermSpec['norm']): string | null {
  if (!norm) return null;
  const labels: Record<NonNullable<LossTermSpec['norm']>, string> = {
    squared_l2: 'Squared L2',
    l2: 'L2',
    l1: 'L1',
    huber: 'Huber',
  };
  return labels[norm];
}

function formatTimeAggregation(timeAgg?: TimeAggregationSpec): string | null {
  if (!timeAgg) return null;
  let base: string;
  switch (timeAgg.mode) {
    case 'all':
      base = 'All steps';
      break;
    case 'final':
      base = 'Final step';
      break;
    case 'range':
      base = `Range ${timeAgg.start ?? '?'}–${timeAgg.end ?? '?'}`;
      break;
    case 'segment':
      base = `Segment ${timeAgg.segment_name ?? 'unknown'}`;
      break;
    case 'custom':
      base = `Custom steps ${timeAgg.time_idxs?.join(', ') ?? '[]'}`;
      break;
    default:
      base = 'All steps';
      break;
  }
  if (timeAgg.discount && timeAgg.discount !== 'none') {
    if (timeAgg.discount === 'power') {
      return `${base}, power discount (exp=${timeAgg.discount_exp ?? 1})`;
    }
    return `${base}, linear discount`;
  }
  return base;
}

function formatWeight(value: number): string {
  if (!Number.isFinite(value)) return '0.0';
  const fixed = value.toFixed(3);
  const trimmed = fixed.replace(/\.?0+$/, '');
  return trimmed.includes('.') ? trimmed : `${trimmed}.0`;
}

function sanitizeSymbol(label: string): string {
  const cleaned = label
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/^_+|_+$/g, '');
  return cleaned.length > 0 ? cleaned : 'term';
}

// ---------------------------------------------------------------------------
// Training config summary
// ---------------------------------------------------------------------------

interface TrainingConfigSummaryProps {
  networkType: string;
  hiddenDim: number;
  nBatches: number;
  batchSize: number;
  learningRate: number;
}

/**
 * Read-only summary of what will be sent to the backend when Start is clicked.
 * Gives the user visual confirmation that Studio has parsed their graph.
 */
function TrainingConfigSummary({
  networkType,
  hiddenDim,
  nBatches,
  batchSize,
  learningRate,
}: TrainingConfigSummaryProps) {
  const chips: { label: string; value: string }[] = [
    { label: 'net', value: networkType },
    { label: 'dim', value: String(hiddenDim) },
    { label: 'batches', value: String(nBatches) },
    { label: 'batch size', value: String(batchSize) },
    { label: 'lr', value: learningRate.toExponential(1) },
  ];

  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50/70 p-3 space-y-1.5">
      <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Config summary</div>
      <div className="flex flex-wrap gap-1.5">
        {chips.map(({ label, value }) => (
          <span
            key={label}
            className="inline-flex items-baseline gap-1 rounded-md bg-white border border-slate-200 px-2 py-0.5 text-[11px]"
          >
            <span className="text-slate-400">{label}</span>
            <span className="font-mono font-semibold text-slate-700">{value}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

function CloudStatusChip({ status }: { status: string }) {
  const cfg: Record<string, { label: string; className: string; spinner?: boolean }> = {
    idle: { label: 'idle', className: 'bg-slate-100 text-slate-500' },
    creating: { label: 'creating', className: 'bg-amber-100 text-amber-700', spinner: true },
    connecting: { label: 'connecting', className: 'bg-amber-100 text-amber-700', spinner: true },
    running: { label: 'running', className: 'bg-emerald-100 text-emerald-700' },
    preempted: { label: 'preempted', className: 'bg-red-100 text-red-700' },
    error: { label: 'error', className: 'bg-red-100 text-red-700' },
  };
  const { label, className, spinner } = cfg[status] ?? cfg.idle;
  return (
    <span
      className={clsx(
        'flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium',
        className
      )}
    >
      {spinner && <Loader2 className="w-2.5 h-2.5 animate-spin" />}
      {label}
    </span>
  );
}
