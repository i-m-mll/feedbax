import type {
  StudioScenarioEntity,
  StudioScenarioEntityKind,
  StudioScenarioEntityRegistry,
} from '@/types/workspace';

export interface ScenarioProjectionItem {
  entity_id: string;
  kind: StudioScenarioEntityKind;
  label: string;
  summary: string | null;
  related_entity_ids: string[];
}

function itemFromEntity(entity: StudioScenarioEntity): ScenarioProjectionItem {
  return {
    entity_id: entity.id,
    kind: entity.kind,
    label: entity.label,
    summary: entity.summary ?? null,
    related_entity_ids: entity.relations.map((relation) => relation.entity_id),
  };
}

export function workspaceProjectionItems(
  registry: StudioScenarioEntityRegistry
): ScenarioProjectionItem[] {
  const primaryKinds = new Set<StudioScenarioEntityKind>([
    'task_object',
    'task_output',
    'task_binding',
    'mechanics_object',
  ]);
  const objectiveItems = Object.values(registry.entities)
    .filter((entity) => entity.kind === 'objective_term')
    .filter((entity) =>
      entity.relations.some((relation) => {
        const related = registry.entities[relation.entity_id];
        return related?.kind === 'graph_port' || related?.kind === 'mechanics_object';
      })
    );

  return [
    ...Object.values(registry.entities).filter((entity) => primaryKinds.has(entity.kind)),
    ...objectiveItems,
  ].map(itemFromEntity);
}

export function objectiveProjectionItems(
  registry: StudioScenarioEntityRegistry
): ScenarioProjectionItem[] {
  return Object.values(registry.entities)
    .filter((entity) => entity.kind === 'objective_term')
    .sort((a, b) => a.label.localeCompare(b.label))
    .map(itemFromEntity);
}

export function relatedProjectionItems(
  registry: StudioScenarioEntityRegistry,
  entityId: string | null | undefined
): ScenarioProjectionItem[] {
  if (!entityId) return [];
  const entity = registry.entities[entityId];
  if (!entity) return [];
  const relatedIds = new Set(entity.relations.map((relation) => relation.entity_id));
  return Object.values(registry.entities)
    .filter((candidate) => {
      if (relatedIds.has(candidate.id)) return true;
      return candidate.relations.some((relation) => relation.entity_id === entity.id);
    })
    .map(itemFromEntity);
}
