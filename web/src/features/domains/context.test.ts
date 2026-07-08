import { describe, expect, it } from 'vitest';
import { buildDomainContexts } from './context';
import type { ComponentDefinition } from '@/types/components';
import type { DomainMeta } from '@/generated/studioContracts';

const CAUSAL_DOMAIN_ID = 'feedbax.domain.causal';
const ACAUSAL_DOMAIN_ID = 'feedbax.domain.acausal';

const domains: DomainMeta[] = [
  {
    id: CAUSAL_DOMAIN_ID,
    display_name: 'Causal',
    interior_schema_id: 'feedbax.spec.graph',
    edge_semantics: 'directed',
    allows_multi_edge_per_port: false,
    nestable_domains: [CAUSAL_DOMAIN_ID, ACAUSAL_DOMAIN_ID],
    editor: { kind: 'canvas', editable: true },
    theme: { color: 'causal', icon: 'Layers', edge_style: 'directed' },
    compiler_id: null,
  },
  {
    id: ACAUSAL_DOMAIN_ID,
    display_name: 'Acausal',
    interior_schema_id: 'feedbax.spec.acausal_graph',
    edge_semantics: 'undirected',
    allows_multi_edge_per_port: true,
    nestable_domains: [ACAUSAL_DOMAIN_ID],
    editor: { kind: 'canvas', editable: true },
    theme: { color: 'acausal', icon: 'Cog', edge_style: 'undirected' },
    compiler_id: 'feedbax.compiler.acausal',
  },
];

function component(
  name: string,
  domain: string,
  interiorDomain: string | null = null
): ComponentDefinition {
  return {
    name,
    category: 'Structure',
    description: name,
    param_schema: [],
    input_ports: [],
    output_ports: [],
    icon: 'Layers',
    default_params: {},
    domain,
    interior_domain: interiorDomain,
    is_composite: Boolean(interiorDomain),
  };
}

describe('domain contexts', () => {
  it('filters the palette by placement domain', () => {
    const contexts = buildDomainContexts(domains);
    const acausal = contexts.get(ACAUSAL_DOMAIN_ID)!;

    expect(acausal.paletteFilter(component('Spring', ACAUSAL_DOMAIN_ID))).toBe(true);
    expect(acausal.paletteFilter(component('Subgraph', CAUSAL_DOMAIN_ID, CAUSAL_DOMAIN_ID))).toBe(false);
  });

  it('rejects causal Subgraph inside acausal interiors with a domain-rule verdict', () => {
    const contexts = buildDomainContexts(domains);
    const acausal = contexts.get(ACAUSAL_DOMAIN_ID)!;

    const verdict = acausal.canPlace(
      component('Subgraph', CAUSAL_DOMAIN_ID, CAUSAL_DOMAIN_ID)
    );

    expect(verdict.allowed).toBe(false);
    expect('reason' in verdict ? verdict.reason : '').toContain(
      "Acausal interiors accept acausal-domain components only; 'Subgraph' is causal."
    );
  });

  it('allows causal components that declare an acausal interior from the causal layer', () => {
    const contexts = buildDomainContexts(domains);
    const causal = contexts.get(CAUSAL_DOMAIN_ID)!;

    expect(causal.canPlace(component('AcausalSystem', CAUSAL_DOMAIN_ID, ACAUSAL_DOMAIN_ID))).toEqual({
      allowed: true,
    });
  });
});
