import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, CheckCircle2, Loader2 } from 'lucide-react';
import { compilePenzaiNode, inspectPenzaiNode } from '@/api/client';
import type {
  DomainCompileReport,
  DomainDiagnostic,
  PenzaiNodeRequest,
} from '@/generated/studioContracts';
import type { ComponentSpec } from '@/types/graph';

function diagnosticClass(diagnostic: DomainDiagnostic) {
  if (diagnostic.severity === 'error') return 'border-rose-200 bg-rose-50 text-rose-800';
  if (diagnostic.severity === 'warning') return 'border-amber-200 bg-amber-50 text-amber-800';
  return 'border-slate-200 bg-slate-50 text-slate-700';
}

function nodeRequest(nodePath: string[], spec: ComponentSpec): PenzaiNodeRequest | null {
  const builderName = spec.params.builder_name;
  if (typeof builderName !== 'string' || builderName.length === 0) return null;
  const params = Object.fromEntries(
    Object.entries(spec.params).filter(
      ([key]) => !['builder_name', 'input_port', 'output_port'].includes(key)
    )
  );
  return {
    node_path: nodePath,
    builder_name: builderName,
    params,
    input_port: typeof spec.params.input_port === 'string' ? spec.params.input_port : 'input',
    output_port: typeof spec.params.output_port === 'string' ? spec.params.output_port : 'output',
  };
}

export function PenzaiInspector({
  graphId,
  nodePath,
  nodeSpec,
}: {
  graphId: string | null;
  nodePath: string[];
  nodeSpec: ComponentSpec;
}) {
  const request = useMemo(() => nodeRequest(nodePath, nodeSpec), [nodePath, nodeSpec]);
  const [html, setHtml] = useState('');
  const [report, setReport] = useState<DomainCompileReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setHtml('');
    setReport(null);
    setError(null);
    if (!request) {
      setError('Select a registered builder before opening this inspector.');
      return;
    }
    setLoading(true);
    const load = async () => {
      try {
        const compileReport = await compilePenzaiNode(graphId, request);
        if (cancelled) return;
        setReport(compileReport);
        if (compileReport.status === 'error') return;
        const inspected = await inspectPenzaiNode(request);
        if (cancelled) return;
        setHtml(inspected.html);
        setReport(inspected.report);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [graphId, request]);

  return (
    <div className="absolute inset-0 z-10 flex flex-col bg-slate-50">
      <div className="flex min-h-12 items-center justify-between border-b border-slate-200 bg-white px-5">
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold text-slate-800">
            {nodePath[nodePath.length - 1] ?? 'Inspector'}
          </div>
          {request && (
            <div className="text-xs text-slate-500">
              {request.builder_name}
              {report?.status ? ` - ${report.status}` : ''}
            </div>
          )}
        </div>
        {loading ? (
          <Loader2 className="h-4 w-4 animate-spin text-slate-400" aria-label="Loading" />
        ) : report?.status === 'ok' ? (
          <CheckCircle2 className="h-4 w-4 text-emerald-600" aria-label="OK" />
        ) : report?.status === 'error' ? (
          <AlertTriangle className="h-4 w-4 text-rose-600" aria-label="Error" />
        ) : null}
      </div>
      {report?.summary && Object.keys(report.summary).length > 0 && (
        <div className="flex flex-wrap gap-2 border-b border-slate-200 bg-white px-5 py-2 text-xs text-slate-600">
          {Object.entries(report.summary).map(([key, value]) => (
            <span key={key} className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
              {key.replace(/_/g, ' ')}: {value}
            </span>
          ))}
        </div>
      )}
      {report?.diagnostics && report.diagnostics.length > 0 && (
        <div className="space-y-2 border-b border-slate-200 bg-white px-5 py-3">
          {report.diagnostics.map((diagnostic, index) => (
            <div
              key={`${diagnostic.code}-${index}`}
              className={`rounded-md border px-3 py-2 text-xs ${diagnosticClass(diagnostic)}`}
            >
              <span className="font-semibold">{diagnostic.code}</span>
              <span className="ml-2">{diagnostic.message}</span>
            </div>
          ))}
        </div>
      )}
      {error && (
        <div className="m-5 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800">
          {error}
        </div>
      )}
      {html ? (
        <iframe
          title="Structural inspector"
          sandbox=""
          srcDoc={html}
          className="h-full w-full flex-1 border-0 bg-white"
        />
      ) : (
        !error &&
        !loading && (
          <div className="flex flex-1 items-center justify-center px-6 text-sm text-slate-500">
            No structural view is available for this node.
          </div>
        )
      )}
    </div>
  );
}
