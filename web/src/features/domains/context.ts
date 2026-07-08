import type { ComponentDefinition } from '@/types/components';
import type { DomainMeta } from '@/generated/studioContracts';

export type PlacementVerdict =
  | { allowed: true }
  | { allowed: false; reason: string };

export interface DomainContext {
  id: string;
  domain: DomainMeta;
  theme: {
    canvasTintClass: string;
    nodeToneClass: string;
    icon: string;
  };
  paletteFilter: (component: ComponentDefinition) => boolean;
  canPlace: (component: ComponentDefinition) => PlacementVerdict;
  connectionSemantics: 'directed' | 'undirected';
}

const themeClasses: Record<string, Pick<DomainContext['theme'], 'canvasTintClass' | 'nodeToneClass'>> = {
  causal: {
    canvasTintClass: 'bg-violet-50/40',
    nodeToneClass: 'bg-violet-100 text-violet-700 border-violet-200',
  },
  acausal: {
    canvasTintClass: 'bg-teal-50/40',
    nodeToneClass: 'bg-teal-100 text-teal-700 border-teal-200',
  },
  mechanics: {
    canvasTintClass: 'bg-emerald-50/40',
    nodeToneClass: 'bg-emerald-100 text-emerald-700 border-emerald-200',
  },
  penzai: {
    canvasTintClass: 'bg-amber-50/40',
    nodeToneClass: 'bg-amber-100 text-amber-700 border-amber-200',
  },
};

function domainDisplayName(domainId: string | null | undefined, domainsById: Map<string, DomainMeta>) {
  if (!domainId) return 'unknown';
  return domainsById.get(domainId)?.display_name ?? domainId.replace(/^feedbax\.domain\./, '');
}

function quotedComponentName(component: ComponentDefinition) {
  return `'${component.name}'`;
}

export function buildDomainContexts(domains: DomainMeta[]): Map<string, DomainContext> {
  const domainsById = new Map(domains.map((domain) => [domain.id, domain]));

  return new Map(
    domains.map((domain) => {
      const theme = themeClasses[domain.theme.color] ?? {
        canvasTintClass: 'bg-slate-50/40',
        nodeToneClass: 'bg-slate-100 text-slate-700 border-slate-200',
      };
      const context: DomainContext = {
        id: domain.id,
        domain,
        theme: {
          ...theme,
          icon: domain.theme.icon,
        },
        paletteFilter: (component) => component.domain === domain.id,
        canPlace: (component) => {
          if (component.domain !== domain.id) {
            const accepted = domain.display_name.toLowerCase();
            const actual = domainDisplayName(component.domain, domainsById).toLowerCase();
            return {
              allowed: false,
              reason:
                `${domain.display_name} interiors accept ${accepted}-domain components only; ` +
                `${quotedComponentName(component)} is ${actual}.`,
            };
          }
          const interiorDomain = component.interior_domain ?? null;
          if (interiorDomain && !domain.nestable_domains.includes(interiorDomain)) {
            return {
              allowed: false,
              reason:
                `${domain.display_name} interiors cannot contain ` +
                `${domainDisplayName(interiorDomain, domainsById).toLowerCase()} interiors.`,
            };
          }
          return { allowed: true };
        },
        connectionSemantics: domain.edge_semantics,
      };
      return [domain.id, context];
    })
  );
}

export function resolveDomainContext(
  contextId: string,
  contexts: Map<string, DomainContext>,
  defaultDomainId?: string | null
): DomainContext | null {
  return contexts.get(contextId) ?? (defaultDomainId ? contexts.get(defaultDomainId) ?? null : null);
}
