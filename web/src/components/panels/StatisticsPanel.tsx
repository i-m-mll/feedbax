import { useCallback, useEffect, useMemo } from 'react';
import clsx from 'clsx';
import { Loader2 } from 'lucide-react';
import { useTrajectoryStore } from '@/stores/trajectoryStore';
import { useStatisticsStore } from '@/stores/statisticsStore';
import { METRIC_LABELS, GROUP_BY_OPTIONS } from '@/types/statistics';
import { MetricCard } from '@/components/charts/MetricCard';
import { TimeseriesChart } from '@/components/charts/TimeseriesChart';
import { HistogramChart } from '@/components/charts/HistogramChart';
import { ScatterPlotChart } from '@/components/charts/ScatterChart';
import { DiagnosticsPanel } from '@/components/charts/DiagnosticsPanel';

const METRIC_OPTIONS = Object.entries(METRIC_LABELS).map(([value, label]) => ({
  value,
  label,
}));

export function StatisticsPanel() {
  const {
    datasets,
    activeDataset,
    loadDatasets,
    selectDataset,
  } = useTrajectoryStore();

  const {
    groupBy,
    selectedMetric,
    scatterXMetric,
    scatterYMetric,
    summaryData,
    timeseriesData,
    histogramData,
    scatterData,
    diagnosticsData,
    loading,
    error,
    activeSubTab,
    activeChartSubTab,
    setGroupBy,
    setSelectedMetric,
    setScatterMetrics,
    setActiveSubTab,
    setActiveChartSubTab,
    loadSummary,
    loadTimeseries,
    loadHistogram,
    loadScatter,
    loadDiagnostics,
  } = useStatisticsStore();

  // Load datasets on mount
  useEffect(() => {
    loadDatasets();
  }, [loadDatasets]);

  // Reload summary when groupBy changes (if dataset is active)
  useEffect(() => {
    if (activeDataset) {
      loadSummary();
    }
  }, [groupBy, activeDataset, loadSummary]);

  // Load timeseries for overview tab when summary is loaded
  useEffect(() => {
    if (activeSubTab === 'overview' && activeDataset && !timeseriesData) {
      loadTimeseries();
    }
  }, [activeSubTab, activeDataset, timeseriesData, loadTimeseries]);

  // Load chart data when chart sub-tab changes
  useEffect(() => {
    if (activeSubTab !== 'charts' || !activeDataset) return;
    if (activeChartSubTab === 'timeseries' && !timeseriesData) loadTimeseries();
    if (activeChartSubTab === 'histogram' && !histogramData) loadHistogram();
    if (activeChartSubTab === 'scatter' && !scatterData) loadScatter();
  }, [
    activeSubTab,
    activeChartSubTab,
    activeDataset,
    timeseriesData,
    histogramData,
    scatterData,
    loadTimeseries,
    loadHistogram,
    loadScatter,
  ]);

  // Load diagnostics when switching to diagnostics tab
  useEffect(() => {
    if (activeSubTab === 'diagnostics' && activeDataset && !diagnosticsData) {
      loadDiagnostics();
    }
  }, [activeSubTab, activeDataset, diagnosticsData, loadDiagnostics]);

  const handleDatasetChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = e.target.value;
      if (value) selectDataset(value);
    },
    [selectDataset],
  );

  const handleGroupByChange = useCallback(
    (value: string) => {
      setGroupBy(value);
    },
    [setGroupBy],
  );

  const handleMetricChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setSelectedMetric(e.target.value);
      // Clear cached chart data so it reloads with new metric
      useStatisticsStore.setState({
        timeseriesData: null,
        histogramData: null,
      });
    },
    [setSelectedMetric],
  );

  const handleScatterXChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setScatterMetrics(e.target.value, scatterYMetric);
      useStatisticsStore.setState({ scatterData: null });
    },
    [setScatterMetrics, scatterYMetric],
  );

  const handleScatterYChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      setScatterMetrics(scatterXMetric, e.target.value);
      useStatisticsStore.setState({ scatterData: null });
    },
    [setScatterMetrics, scatterXMetric],
  );

  // Extract metric names from summary data for metric cards
  const metricKeys = useMemo(() => {
    if (!summaryData?.groups.length) return [];
    // Use first group's metrics as canonical set
    return Object.keys(summaryData.groups[0].metrics);
  }, [summaryData]);

  // For overview, show aggregated metrics (first group if groupBy=none, or all groups)
  const overviewMetrics = useMemo(() => {
    if (!summaryData?.groups.length) return null;
    // When not grouping, show the single group's metrics
    if (summaryData.group_by === 'none' && summaryData.groups.length === 1) {
      return summaryData.groups[0].metrics;
    }
    // When grouping, show the first group (user can see per-group in charts)
    return summaryData.groups[0].metrics;
  }, [summaryData]);

  return (
    <div className="flex flex-col h-full">
      {/* Top bar: dataset selector + group by */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-100 bg-white/80 flex-shrink-0 flex-wrap">
        {/* Dataset selector */}
        <select
          value={activeDataset ?? ''}
          onChange={handleDatasetChange}
          className="rounded-lg border border-slate-200 px-2 py-1 text-sm text-slate-700 bg-white min-w-[140px]"
        >
          <option value="">Select dataset...</option>
          {datasets.map((ds) => (
            <option key={ds.name} value={ds.name}>
              {ds.name}
            </option>
          ))}
        </select>

        {/* Group by buttons */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-slate-400 mr-1">Group by:</span>
          {GROUP_BY_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => handleGroupByChange(opt.value)}
              className={clsx(
                'text-[10px] font-semibold px-2 py-0.5 rounded-full border',
                groupBy === opt.value
                  ? 'border-brand-500 text-brand-600 bg-brand-500/10'
                  : 'border-slate-200 text-slate-400 hover:text-slate-600 hover:bg-slate-50',
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Sub-tabs */}
        <div className="flex items-center gap-1">
          {(['overview', 'charts', 'diagnostics'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveSubTab(tab)}
              className={clsx(
                'text-[10px] font-semibold px-2.5 py-0.5 rounded-full border capitalize',
                activeSubTab === tab
                  ? 'border-brand-500 text-brand-600 bg-brand-500/10'
                  : 'border-transparent text-slate-400 hover:text-slate-600 hover:bg-slate-50',
              )}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Content area */}
      <div className="flex-1 relative min-h-0 overflow-y-auto">
        {/* Error banner */}
        {error && (
          <div className="absolute top-2 left-2 right-2 z-10 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-xs text-red-600">
            {error}
          </div>
        )}

        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/60">
            <Loader2 className="w-6 h-6 text-brand-500 animate-spin" />
          </div>
        )}

        {/* Empty state */}
        {!activeDataset && !loading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-sm text-slate-400">Select a dataset to view statistics</div>
          </div>
        )}

        {/* Overview tab */}
        {activeDataset && activeSubTab === 'overview' && (
          <div className="p-4 space-y-4">
            {/* Metric cards row */}
            {overviewMetrics && (
              <div className="flex gap-3 overflow-x-auto pb-1">
                {metricKeys.map((key) => (
                  <MetricCard
                    key={key}
                    label={METRIC_LABELS[key] ?? key}
                    summary={overviewMetrics[key]}
                  />
                ))}
              </div>
            )}

            {/* Default timeseries chart */}
            {timeseriesData && (
              <div className="h-[240px]">
                <TimeseriesChart data={timeseriesData} />
              </div>
            )}
          </div>
        )}

        {/* Charts tab */}
        {activeDataset && activeSubTab === 'charts' && (
          <div className="p-4 space-y-3">
            {/* Chart toolbar */}
            <div className="flex items-center gap-3 flex-wrap">
              {/* Chart sub-tabs */}
              <div className="flex items-center gap-1">
                {(['timeseries', 'histogram', 'scatter'] as const).map((ct) => (
                  <button
                    key={ct}
                    onClick={() => setActiveChartSubTab(ct)}
                    className={clsx(
                      'text-[10px] font-semibold px-2 py-0.5 rounded-full border capitalize',
                      activeChartSubTab === ct
                        ? 'border-brand-500 text-brand-600 bg-brand-500/10'
                        : 'border-slate-200 text-slate-400 hover:text-slate-600 hover:bg-slate-50',
                    )}
                  >
                    {ct}
                  </button>
                ))}
              </div>

              {/* Metric selector (for timeseries and histogram) */}
              {(activeChartSubTab === 'timeseries' || activeChartSubTab === 'histogram') && (
                <select
                  value={selectedMetric}
                  onChange={handleMetricChange}
                  className="rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-700 bg-white"
                >
                  {METRIC_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              )}

              {/* Scatter axis selectors */}
              {activeChartSubTab === 'scatter' && (
                <>
                  <span className="text-[10px] text-slate-400">X:</span>
                  <select
                    value={scatterXMetric}
                    onChange={handleScatterXChange}
                    className="rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-700 bg-white"
                  >
                    {METRIC_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                  <span className="text-[10px] text-slate-400">Y:</span>
                  <select
                    value={scatterYMetric}
                    onChange={handleScatterYChange}
                    className="rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-700 bg-white"
                  >
                    {METRIC_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </>
              )}
            </div>

            {/* Chart area */}
            <div className="h-[280px]">
              {activeChartSubTab === 'timeseries' && timeseriesData && (
                <TimeseriesChart data={timeseriesData} />
              )}
              {activeChartSubTab === 'histogram' && histogramData && (
                <HistogramChart data={histogramData} />
              )}
              {activeChartSubTab === 'scatter' && scatterData && (
                <ScatterPlotChart data={scatterData} />
              )}
              {!loading &&
                ((activeChartSubTab === 'timeseries' && !timeseriesData) ||
                  (activeChartSubTab === 'histogram' && !histogramData) ||
                  (activeChartSubTab === 'scatter' && !scatterData)) && (
                  <div className="h-full flex items-center justify-center text-sm text-slate-400">
                    No data loaded
                  </div>
                )}
            </div>
          </div>
        )}

        {/* Diagnostics tab */}
        {activeDataset && activeSubTab === 'diagnostics' && (
          <div className="p-4">
            {diagnosticsData ? (
              <DiagnosticsPanel data={diagnosticsData} />
            ) : (
              !loading && (
                <div className="text-sm text-slate-400">No diagnostics loaded</div>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
