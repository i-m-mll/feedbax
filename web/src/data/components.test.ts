import { describe, expect, it } from 'vitest';
import { componentLibrary } from '@/data/components';
import { parseContract } from '@/generated/studioContracts';

describe('component registry fallback', () => {
  it('does not carry a hand-maintained offline registry', () => {
    expect(componentLibrary).toEqual([]);
  });

  it('parses representation contracts from component catalog responses', () => {
    const response = parseContract('ComponentListResponse', {
      schema_id: 'feedbax.spec.studio.api_transport',
      schema_version: 'feedbax.spec.studio.api_transport.v2',
      data: {
        schema_id: 'feedbax.spec.studio.api_transport',
        schema_version: 'feedbax.spec.studio.api_transport.v2',
        components: [
          {
            name: 'RepresentedGain',
            category: 'Test',
            description: 'Represented component fixture',
            param_schema: [{ name: 'gain', type: 'float', default: 1.0, required: false }],
            input_ports: ['input'],
            output_ports: ['output'],
            icon: 'box',
            default_params: { gain: 1.0 },
            param_schema_version: '1',
            supported_param_schema_versions: ['1'],
            migrations: [],
            trainable_by_default: false,
            representation: {
              schema_id: 'feedbax.spec.studio.representation',
              schema_version: 'feedbax.spec.studio.representation.v4',
              anchors: [
                {
                  id: 'endpoint',
                  semantic_role: 'endpoint',
                  interaction_roles: ['selectable'],
                  binding: { kind: 'param_path', path: 'gain' },
                },
              ],
              elements: [
                {
                  id: 'marker',
                  archetype: 'marker',
                  anchors: ['endpoint'],
                  bindings: {},
                  style: [],
                  scale_invariant: false,
                  metadata: {},
                },
              ],
              style: [],
              scale_invariant: false,
              metadata: {},
            },
          },
        ],
      },
    });

    expect(response.data.components[0].representation?.elements[0].archetype).toBe('marker');
  });
});
