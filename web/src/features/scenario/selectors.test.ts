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
import type { StudioSchemaRegistry } from '@/types/workspace';

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

function schemaRegistry(): StudioSchemaRegistry {
  return {
    kind: 'studio_schema_registry',
    schema_version: 'feedbax.studio.v1',
    generated_at: '2026-05-20T00:00:00Z',
    workspace_id: 'workspace:test',
    scenario_id: 'scenario:test',
    ports: [
      {
        id: 'port:decoder.readout:output',
        label: 'decoder.readout',
        node_id: 'decoder',
        component_type: 'Decoder',
        port: 'readout',
        direction: 'output',
        value_schema: {
          id: 'value:port:decoder.readout:output',
          label: 'decoder.readout',
          kind: 'graph_port',
          dtype: 'float32',
          shape: ['time', 4],
          rank: 2,
          units: 'a.u.',
          frame: null,
          origin: 'declared',
          metadata: {},
        },
        bound_task_data_id: null,
        origin: 'declared',
        metadata: {},
      },
    ],
    task_data: [],
    selector_targets: [
      {
        id: 'selector:path:states.decoder.readout',
        label: 'Decoder readout',
        kind: 'state_hint',
        selector: 'path:states.decoder.readout',
        value_schema: {
          id: 'value:path:states.decoder.readout',
          label: 'Decoder readout',
          kind: 'trajectory',
          dtype: 'float32',
          shape: ['time', 4],
          rank: 2,
          units: 'a.u.',
          frame: null,
          origin: 'declared',
          metadata: {},
        },
        origin: 'declared',
        source: {
          graph_port_node_id: 'decoder',
          graph_port_name: 'readout',
          graph_port_direction: 'output',
        },
        metadata: { detail: 'provider schema' },
      },
      {
        id: 'selector:edge:decoder.readout->mechanics.force',
        label: 'decoder.readout -> mechanics.force',
        kind: 'edge',
        selector: 'edge:decoder.readout->mechanics.force',
        value_schema: {
          id: 'value:edge:decoder.readout->mechanics.force',
          label: 'decoder.readout',
          kind: 'graph_edge',
          dtype: 'float32',
          shape: ['time', 4],
          rank: 2,
          units: 'a.u.',
          frame: null,
          origin: 'declared',
          metadata: {},
        },
        origin: 'declared',
        source: {
          source_node: 'decoder',
          source_port: 'readout',
          target_node: 'mechanics',
          target_port: 'force',
        },
        metadata: { default_retention: { mode: 'trajectory' } },
      },
      {
        id: 'selector:graph_output:effector',
        label: 'Graph output effector',
        kind: 'graph_output',
        selector: 'graph_output:effector',
        value_schema: {
          id: 'value:graph_output:effector',
          label: 'effector',
          kind: 'graph_output',
          dtype: 'float32',
          shape: ['time', 2],
          rank: 2,
          units: 'cm',
          frame: null,
          origin: 'declared',
          metadata: {},
        },
        origin: 'declared',
        source: {
          output_name: 'effector',
          node_id: 'mechanics',
          port: 'effector',
        },
        metadata: { default_retention: { mode: 'trajectory' } },
      },
    ],
    issues: [],
    metadata: {},
  };
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

  it('derives selector choices from provider schema targets when available', () => {
    const options = selectorOptionsForRegistry({
      registry: registry(),
      schemaRegistry: schemaRegistry(),
    });
    const option = options.find(
      (candidate) => candidate.selector.compact === 'path:states.decoder.readout'
    );

    expect(option).toMatchObject({
      group: 'state',
      label: 'Decoder readout',
      detail: 'a.u. · float32 · time x 4 · provider schema',
      origin: 'declared',
      schema_target_id: 'selector:path:states.decoder.readout',
      selector: expect.objectContaining({
        namespace: 'state_path',
        expected_shape: ['time', 4],
        dtype: 'float32',
        units: 'a.u.',
        metadata: expect.objectContaining({
          source: 'studio_schema_registry',
          schema_origin: 'declared',
          graph_port_node_id: 'decoder',
          graph_port_name: 'readout',
          graph_port_direction: 'output',
        }),
      }),
    });
    expect(options.some((candidate) => candidate.label === 'Effector position')).toBe(false);
    expect(
      selectorOptionsForGraphPort(
        {
          namespace: 'graph_port',
          compact: 'port:decoder.readout',
          target_id: 'decoder',
          path: 'readout',
          role: 'observed',
          metadata: { direction: 'output' },
        },
        options
      ).map((candidate) => candidate.label)
    ).toEqual(['Decoder readout']);
  });

  it('represents graph structural selector target kinds from provider schema', () => {
    const options = selectorOptionsForRegistry({
      registry: registry(),
      schemaRegistry: schemaRegistry(),
    });

    expect(options).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          group: 'observables',
          label: 'decoder.readout -> mechanics.force',
          selector: expect.objectContaining({
            namespace: 'graph_edge',
            compact: 'edge:decoder.readout->mechanics.force',
            target_id: 'decoder:readout->mechanics:force',
          }),
        }),
        expect.objectContaining({
          group: 'observables',
          label: 'Graph output effector',
          selector: expect.objectContaining({
            namespace: 'graph_output',
            compact: 'graph_output:effector',
            target_id: 'effector',
            metadata: expect.objectContaining({
              graph_port_node_id: 'mechanics',
              graph_port_name: 'effector',
            }),
          }),
        }),
      ])
    );
  });

  it('uses curated state hints as explicit fallback entries without provider schema targets', () => {
    const option = selectorOptionsForRegistry({ registry: registry() }).find(
      (candidate) => candidate.selector.compact === 'path:states.mechanics.effector.pos'
    );

    expect(option).toMatchObject({
      origin: 'state_browser',
      selector: expect.objectContaining({
        metadata: expect.objectContaining({
          source: 'curated_state_hint',
          schema_origin: 'curated_fallback',
          graph_port_node_id: 'mechanics',
          graph_port_name: 'effector',
        }),
      }),
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
