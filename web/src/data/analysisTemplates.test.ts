import { afterEach, describe, expect, it } from 'vitest';
import { useAnalysisStore } from '@/stores/analysisStore';
import { createRlrmpPart1Analysis } from './rlrmp-part1';
import { createRlrmpPart2Project } from './rlrmp-part2';
import { createRlrmpMovementRampAnalysis } from './rlrmp-run-example';

const initialAnalysisState = useAnalysisStore.getState();

afterEach(() => useAnalysisStore.setState(initialAnalysisState, true));

describe.each([
  ['RLRMP Part 1', createRlrmpPart1Analysis],
  ['RLRMP Part 2', createRlrmpPart2Project],
  ['RLRMP movement ramp', createRlrmpMovementRampAnalysis],
] as const)('%s template', (_name, createSnapshot) => {
  it('opens every page with resolvable canvas connections and preserves its authored graph', () => {
    const snapshot = JSON.parse(JSON.stringify(createSnapshot())) as ReturnType<typeof createSnapshot>;
    expect(snapshot.pages.length).toBeGreaterThan(0);
    expect(snapshot.pages.some((page) => page.id === snapshot.activePageId)).toBe(true);
    expect(new Set(snapshot.pages.map((page) => page.id)).size).toBe(snapshot.pages.length);

    const nodeIds: string[] = [];
    const wireIds: string[] = [];
    for (const page of snapshot.pages) {
      const { nodes, wires, dataSourceId } = page.graphSpec;
      nodeIds.push(...Object.keys(nodes));
      wireIds.push(...wires.map((wire) => wire.id));
      for (const [id, node] of Object.entries(nodes)) expect(node.id).toBe(id);
      for (const wire of wires) {
        expect(nodes[wire.targetId]?.inputPorts, wire.id).toContain(wire.targetPort);
        if (wire.sourceId !== dataSourceId) {
          expect(nodes[wire.sourceId]?.outputPorts, wire.id).toContain(wire.sourcePort);
        }
        if (wire.implicit || wire.fieldPath) expect(wire.sourceId).toBe(dataSourceId);
      }

      useAnalysisStore.getState().restoreSnapshot({ ...snapshot, activePageId: page.id });
      const rendered = useAnalysisStore.getState();
      const renderedIds = new Set(rendered.nodes.map((node) => node.id));
      expect(renderedIds.has(dataSourceId)).toBe(true);
      for (const id of Object.keys(nodes)) expect(renderedIds.has(id)).toBe(true);
      for (const edge of rendered.edges) {
        expect(renderedIds.has(edge.source), edge.id).toBe(true);
        expect(renderedIds.has(edge.target), edge.id).toBe(true);
      }
      for (const wire of wires) {
        if (wire.transform) {
          expect(renderedIds.has(wire.transform.id)).toBe(true);
          expect(rendered.edges).toEqual(expect.arrayContaining([
            expect.objectContaining({ source: wire.sourceId, target: wire.transform.id }),
            expect.objectContaining({ source: wire.transform.id, target: wire.targetId }),
          ]));
        } else {
          expect(rendered.edges).toContainEqual(expect.objectContaining({
            source: wire.sourceId,
            sourceHandle: wire.fieldPath ?? wire.sourcePort,
            target: wire.targetId,
            targetHandle: wire.targetPort,
          }));
        }
      }
      const savedPage = rendered.captureSnapshot().pages.find((saved) => saved.id === page.id)!;
      expect(savedPage.graphSpec).toEqual(page.graphSpec);
      expect(savedPage.evalParams).toEqual(page.evalParams);
    }
    expect(new Set(nodeIds).size).toBe(nodeIds.length);
    expect(new Set(wireIds).size).toBe(wireIds.length);
  });
});
