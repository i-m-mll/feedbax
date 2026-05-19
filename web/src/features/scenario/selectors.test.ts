import { describe, expect, it } from 'vitest';
import { createRlrmpModelGraph } from '@/data/rlrmp-model-graph';
import { buildScenarioEntityRegistry } from '@/features/scenario/entities';
import {
  preferredSelectorForGraphPort,
  selectorAccessExpression,
  selectorDisplayLabel,
  selectorOptionsForGraphPort,
  selectorOptionsForRegistry,
  selectorWithAccessExpression,
} from '@/features/scenario/selectors';
import { buildWorkspaceSnapshot } from '@/stores/workspaceStore';
import { defaultTaskSpec, defaultTrainingSpec } from '@/stores/trainingStore';

function registry() {
  const { graph, uiState } = createRlrmpModelGraph('selector test');
  const workspace = buildWorkspaceSnapshot({
    workspace: null,
    graph,
    uiState,
    trainingSpec: defaultTrainingSpec,
    taskSpec: defaultTaskSpec,
    analysisSnapshot: null,
  });
  const trainStage = workspace.stages.find((stage) => stage.kind === 'train');
  const scenario = trainStage?.scenario_id ? workspace.scenarios[trainStage.scenario_id] : null;
  return buildScenarioEntityRegistry({ scenario, graph });
}

describe('scenario selector options', () => {
  it('includes graph ports and semantic state paths as selector choices', () => {
    const options = selectorOptionsForRegistry({ registry: registry() });

    expect(options).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          group: 'ports',
          selector: expect.objectContaining({ compact: 'port:mechanics.effector' }),
        }),
        expect.objectContaining({
          group: 'state',
          label: 'Effector position',
          selector: expect.objectContaining({
            compact: 'path:states.mechanics.effector.pos',
            metadata: expect.objectContaining({
              graph_port_node_id: 'mechanics',
              graph_port_name: 'effector',
              subpath: 'position',
            }),
          }),
        }),
        expect.objectContaining({
          group: 'state',
          label: 'Network hidden state',
          selector: expect.objectContaining({
            compact: 'path:states.net.hidden',
          }),
        }),
      ])
    );
  });

  it('uses semantic labels instead of compact selector strings when possible', () => {
    const option = selectorOptionsForRegistry({ registry: registry() }).find(
      (candidate) => candidate.selector.compact === 'path:states.mechanics.effector.vel'
    );

    expect(option).toBeDefined();
    expect(selectorDisplayLabel(option?.selector)).toBe('Effector velocity');
  });

  it('offers semantic subfields for graph ports before falling back to the raw port', () => {
    const options = selectorOptionsForRegistry({ registry: registry() });
    const portSelector = {
      namespace: 'graph_port' as const,
      compact: 'port:mechanics.effector',
      target_id: 'mechanics',
      path: 'effector',
      role: 'observed' as const,
      metadata: { direction: 'output' },
    };

    expect(selectorOptionsForGraphPort(portSelector, options).map((option) => option.label)).toEqual(
      ['Effector position', 'Effector velocity']
    );
    expect(preferredSelectorForGraphPort(portSelector, options)).toMatchObject({
      compact: 'path:states.mechanics.effector.pos',
    });
  });

  it('represents arbitrary PyTree and array sub-selections relative to any selector', () => {
    const baseSelector = {
      namespace: 'graph_port' as const,
      compact: 'port:feedback.output',
      target_id: 'feedback',
      path: 'output',
      role: 'observed' as const,
      metadata: { direction: 'output' },
    };

    const selector = selectorWithAccessExpression(baseSelector, '.state["position"][0:2]');

    expect(selector).toMatchObject({
      namespace: 'custom',
      compact: 'port:feedback.output.state["position"][0:2]',
      metadata: expect.objectContaining({
        access_expression: '.state["position"][0:2]',
        base_selector: baseSelector,
      }),
    });
    expect(selectorDisplayLabel(selector)).toBe('feedback.output.state["position"][0:2]');
    expect(selectorAccessExpression(selector)).toBe('.state["position"][0:2]');
    expect(selectorWithAccessExpression(selector, '')).toMatchObject(baseSelector);
  });
});
