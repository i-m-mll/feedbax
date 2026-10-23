import { create } from 'zustand';
import { compileGraphNode } from '@/api/client';
import type { AcausalGraphSpec, DomainCompileReport } from '@/generated/studioContracts';
import { graphPathKey } from '@/features/domains/acausal';

interface CompileStatusState {
  reports: Record<string, DomainCompileReport>;
  compilingPaths: Set<string>;
  setReports: (reports: Record<string, DomainCompileReport> | null | undefined) => void;
  recordReport: (path: string[], report: DomainCompileReport) => void;
  compileNode: (graphId: string, path: string[], interior: AcausalGraphSpec) => Promise<DomainCompileReport>;
}

export const useCompileStatusStore = create<CompileStatusState>((set) => ({
  reports: {},
  compilingPaths: new Set<string>(),
  setReports: (reports) => set({ reports: reports ?? {} }),
  recordReport: (path, report) =>
    set((state) => ({
      reports: {
        ...state.reports,
        [graphPathKey(path)]: report,
      },
    })),
  compileNode: async (graphId, path, interior) => {
    const key = graphPathKey(path);
    set((state) => ({ compilingPaths: new Set([...state.compilingPaths, key]) }));
    try {
      const report = await compileGraphNode(graphId, path, interior);
      set((state) => {
        const compilingPaths = new Set(state.compilingPaths);
        compilingPaths.delete(key);
        return {
          reports: {
            ...state.reports,
            [key]: report,
          },
          compilingPaths,
        };
      });
      return report;
    } catch (error) {
      set((state) => {
        const compilingPaths = new Set(state.compilingPaths);
        compilingPaths.delete(key);
        return { compilingPaths };
      });
      throw error;
    }
  },
}));
