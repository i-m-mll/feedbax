import type {
  GraphSpec,
  ComponentSpec,
} from '@/types/graph';
import type { ComponentDefinition } from '@/types/components';
import type {
  RepresentationElementSpec,
  RepresentationFrameProvider,
  RepresentationLiteralBinding,
  RepresentationParamPathBinding,
  RepresentationSpec,
  RepresentationStateAnchorSelectorBinding,
  RepresentationStyleSpec,
  RepresentationTrialSpecPathBinding,
} from '@/generated/studioContracts';
import {
  mechanicsEntityId,
  objectiveEntityId,
  selectorToEntityId,
  taskEntityId,
} from '@/features/scenario/entities';
import type {
  StudioScenarioEntity,
  StudioScenarioEntityKind,
  StudioScenarioEntityRegistry,
  StudioScenarioSpec,
  StudioSelectorRef,
  StudioObjectiveSpec,
  StudioObjectiveTermSpec,
} from '@/types/workspace';

export interface ScenarioProjectionItem {
  entity_id: string;
  kind: StudioScenarioEntityKind;
  label: string;
  summary: string | null;
  related_entity_ids: string[];
}

export type SceneValidationSeverity = 'warning' | 'error';

export interface ResolvedSceneValidationMessage {
  type: string;
  severity: SceneValidationSeverity;
  message: string;
  entity_id?: string | null;
  path?: string | null;
}

export interface ResolvedSceneAnchor {
  id: string;
  entity_id: string;
  local_id: string;
  label: string;
  semantic_role: string;
  position: [number, number] | null;
  selectable: boolean;
  hoverable: boolean;
  interaction_roles: string[];
  objective_roles: string[];
  frame: string;
  selector: StudioSelectorRef | null;
  metadata: Record<string, unknown>;
}

export type ResolvedSceneGeometry =
  | { kind: 'polyline'; points: Array<[number, number]> }
  | { kind: 'bounds'; min: [number, number]; max: [number, number] }
  | { kind: 'points'; points: Array<[number, number]> }
  | { kind: 'link'; points: Array<[number, number]> }
  | { kind: 'none' };

export interface ResolvedSceneElement {
  id: string;
  entity_id: string;
  local_id: string;
  archetype: string;
  anchor_ids: string[];
  frame: string;
  scale_invariant: boolean;
  style: Record<string, unknown>;
  geometry: ResolvedSceneGeometry;
  metadata: Record<string, unknown>;
}

export interface ResolvedSceneEntity {
  id: string;
  label: string;
  kind: StudioScenarioEntityKind;
  summary: string | null;
  related_entity_ids: string[];
  anchor_ids: string[];
  element_ids: string[];
}

export interface ResolvedScene {
  scenario_id: string | null;
  frame: string;
  units: string;
  entities: ResolvedSceneEntity[];
  anchors: ResolvedSceneAnchor[];
  elements: ResolvedSceneElement[];
  required_selectors: StudioSelectorRef[];
  validation: ResolvedSceneValidationMessage[];
}

type RepresentationBinding =
  | RepresentationParamPathBinding
  | RepresentationStateAnchorSelectorBinding
  | RepresentationTrialSpecPathBinding
  | RepresentationLiteralBinding
  | null
  | undefined;

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
    'task_data',
    'task_binding',
    'mechanics_object',
    'retained_observable',
  ]);
  const objectiveItems = Object.values(registry.entities)
    .filter((entity) => entity.kind === 'objective_term')
    .filter((entity) =>
      entity.relations.some((relation) => {
        const related = registry.entities[relation.entity_id];
        return (
          related?.kind === 'graph_port' ||
          related?.kind === 'graph_edge' ||
          related?.kind === 'mechanics_object' ||
          related?.kind === 'retained_observable' ||
          related?.kind === 'task_data'
        );
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

function componentByName(components: ComponentDefinition[]): Map<string, ComponentDefinition> {
  return new Map(components.map((component) => [component.name, component]));
}

function componentForTask(
  taskType: string,
  componentsByName: Map<string, ComponentDefinition>
): ComponentDefinition | null {
  return (
    componentsByName.get(taskType) ??
    (taskType === 'ReachingTask' ? componentsByName.get('SimpleReaches') : undefined) ??
    null
  );
}

function entityRelations(entity: StudioScenarioEntity): string[] {
  return entity.relations.map((relation) => relation.entity_id);
}

function addUniqueSelector(selectors: StudioSelectorRef[], selector: StudioSelectorRef) {
  if (selectors.some((candidate) => candidate.compact === selector.compact)) return;
  selectors.push(selector);
}

function bindingSelector(binding: RepresentationBinding): StudioSelectorRef | null {
  if (binding?.kind === 'selector') return binding.selector;
  return null;
}

function collectRequiredSelectors(
  selectors: StudioSelectorRef[],
  binding: RepresentationBinding
) {
  const selector = bindingSelector(binding);
  if (selector) addUniqueSelector(selectors, selector);
}

function valueAtPath(source: unknown, path: string): unknown {
  return path.split('.').reduce<unknown>((current, part) => {
    if (current == null) return undefined;
    if (Array.isArray(current)) {
      const index = Number.parseInt(part, 10);
      return Number.isInteger(index) ? current[index] : undefined;
    }
    if (typeof current === 'object') return (current as Record<string, unknown>)[part];
    return undefined;
  }, source);
}

function defaultParam(component: ComponentDefinition, path: string): unknown {
  const explicit = valueAtPath(component.default_params ?? {}, path);
  if (explicit !== undefined) return explicit;
  const root = path.split('.')[0];
  return component.param_schema?.find((param) => param.name === root)?.default;
}

function paramValue(
  component: ComponentDefinition,
  nodeOrTask: ComponentSpec | { params?: Record<string, unknown> | null },
  path: string
): unknown {
  const explicit = valueAtPath(nodeOrTask.params ?? {}, path);
  return explicit === undefined ? defaultParam(component, path) : explicit;
}

function numericPair(value: unknown): [number, number] | null {
  if (!Array.isArray(value) || value.length < 2) return null;
  const x = Number(value[0]);
  const y = Number(value[1]);
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}

function numericArray(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  const parsed = value.map((item) => Number(item));
  return parsed.every(Number.isFinite) ? parsed : null;
}

function numberParamValue(
  component: ComponentDefinition,
  nodeOrTask: ComponentSpec | { params?: Record<string, unknown> | null },
  path: string,
  fallback: number
): number {
  const value = Number(paramValue(component, nodeOrTask, path));
  return Number.isFinite(value) ? value : fallback;
}

function boundsParam(
  component: ComponentDefinition,
  nodeOrTask: ComponentSpec | { params?: Record<string, unknown> | null },
  path: string
): { min: [number, number]; max: [number, number] } | null {
  const value = paramValue(component, nodeOrTask, path);
  if (!Array.isArray(value) || value.length < 2) return null;
  const first = numericPair(value[0]);
  const second = numericPair(value[1]);
  if (!first || !second) return null;
  return {
    min: [Math.min(first[0], second[0]), Math.min(first[1], second[1])],
    max: [Math.max(first[0], second[0]), Math.max(first[1], second[1])],
  };
}

function bindingValue(
  component: ComponentDefinition,
  nodeOrTask: ComponentSpec | { params?: Record<string, unknown> | null },
  binding: RepresentationBinding
): unknown {
  if (!binding) return undefined;
  if (binding.kind === 'literal') return binding.value;
  if (binding.kind === 'param_path') return paramValue(component, nodeOrTask, binding.path);
  return undefined;
}

function frameFromProvider(
  provider: RepresentationFrameProvider | null | undefined,
  fallbackFrame: string,
  graph: GraphSpec | null | undefined,
  nodeId: string | null,
  messages: ResolvedSceneValidationMessage[],
  entityId: string,
  path: string
): string {
  if (!provider) return fallbackFrame;
  if (provider.kind === 'fixed') return provider.frame ?? fallbackFrame;
  if (provider.kind === 'from_input_port') {
    const inputPort = provider.input_port;
    const hasWire =
      Boolean(nodeId && inputPort) &&
      Boolean(
        graph?.wires.some(
          (wire) => wire.target_node === nodeId && wire.target_port === inputPort
        )
      );
    if (!hasWire) {
      messages.push({
        type: 'workspace_unresolved_frame_provider',
        severity: 'warning',
        message: `Frame provider for '${entityId}' requires input port '${inputPort ?? 'unknown'}', but no source frame is available in authoring mode.`,
        entity_id: entityId,
        path,
      });
    }
    return provider.frame ?? fallbackFrame;
  }
  messages.push({
    type: 'workspace_unresolved_frame_provider',
    severity: 'warning',
    message: `Frame provider '${provider.renderer_id ?? provider.kind}' for '${entityId}' is not resolved in authoring mode.`,
    entity_id: entityId,
    path,
  });
  return provider.frame ?? fallbackFrame;
}

function styleMap(styles: RepresentationStyleSpec[] | undefined): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const style of styles ?? []) {
    result[style.channel] = style.value ?? null;
  }
  return result;
}

function anchorGlobalId(entityId: string, anchorId: string): string {
  return `${entityId}::anchor:${anchorId}`;
}

function elementGlobalId(entityId: string, elementId: string): string {
  return `${entityId}::element:${elementId}`;
}

function twoLinkRestPoints(
  component: ComponentDefinition,
  node: ComponentSpec,
  linkLengthPath: string | null
): Array<[number, number]> {
  const lengths = numericArray(
    linkLengthPath ? paramValue(component, node, linkLengthPath) : undefined
  ) ?? [0.3, 0.33];
  const angles =
    numericArray(
      valueAtPath(node.params ?? {}, 'joint_angles') ??
        valueAtPath(node.params ?? {}, 'rest_joint_angles') ??
        valueAtPath(node.params ?? {}, 'initial_angles')
    ) ?? [0, 0];
  const shoulder: [number, number] = [0, 0];
  const theta0 = angles[0] ?? 0;
  const theta1 = theta0 + (angles[1] ?? 0);
  const elbow: [number, number] = [
    Math.cos(theta0) * (lengths[0] ?? 0.3),
    Math.sin(theta0) * (lengths[0] ?? 0.3),
  ];
  const effector: [number, number] = [
    elbow[0] + Math.cos(theta1) * (lengths[1] ?? 0.33),
    elbow[1] + Math.sin(theta1) * (lengths[1] ?? 0.33),
  ];
  return [shoulder, elbow, effector];
}

function taskAuthoringAnchorPosition(
  component: ComponentDefinition,
  task: { params?: Record<string, unknown> | null },
  anchor: { semantic_role: string }
): [number, number] | null {
  const bounds = boundsParam(component, task, 'workspace');
  const center: [number, number] = bounds
    ? [(bounds.min[0] + bounds.max[0]) / 2, (bounds.min[1] + bounds.max[1]) / 2]
    : [0, 0];
  if (anchor.semantic_role === 'origin') return center;
  if (anchor.semantic_role === 'target') {
    return [center[0] + numberParamValue(component, task, 'eval_reach_length', 0.5), center[1]];
  }
  return null;
}

function interactionRoleSet(roles: unknown): Set<string> {
  if (Array.isArray(roles)) return new Set(roles.filter((role) => typeof role === 'string'));
  if (typeof roles === 'string') return new Set([roles]);
  return new Set();
}

function stringList(value: unknown): string[] {
  if (Array.isArray(value)) return value.filter((item): item is string => typeof item === 'string');
  return typeof value === 'string' ? [value] : [];
}

function objectiveRolesForAnchor(
  anchor: { semantic_role: string; interaction_roles?: unknown; metadata?: Record<string, unknown> },
  selector: StudioSelectorRef | null
): string[] {
  const roles = new Set<string>();
  for (const role of stringList(anchor.metadata?.objective_roles)) roles.add(role);
  for (const role of stringList(anchor.metadata?.objective_role)) roles.add(role);
  for (const role of stringList(anchor.metadata?.interaction_roles)) roles.add(role);
  for (const role of stringList(anchor.interaction_roles)) {
    if (
      role.startsWith('objective-') ||
      role.startsWith('canonical-for:') ||
      role === 'illustrative'
    ) {
      roles.add(role);
    }
  }
  const canonicalFor =
    typeof anchor.metadata?.canonical_for === 'string'
      ? anchor.metadata.canonical_for
      : null;
  if (canonicalFor) roles.add(`canonical-for:${canonicalFor}`);
  if (anchor.metadata?.illustrative === true) roles.add('illustrative');
  if (
    selector &&
    (anchor.semantic_role === 'endpoint' || anchor.semantic_role === 'center')
  ) {
    roles.add('objective-source');
  }
  if (anchor.semantic_role === 'target' || anchor.metadata?.canonical_goal === true) {
    roles.add('objective-target');
  }
  return [...roles];
}

function selectorWithRepresentationMetadata(
  selector: StudioSelectorRef | null,
  anchorSubpath: string | null,
  entity: StudioScenarioEntity
): StudioSelectorRef | null {
  if (!selector) return null;
  const nodeId =
    typeof entity.metadata.node_id === 'string' ? entity.metadata.node_id : null;
  const outputPort =
    selector.compact.startsWith('output:') ? selector.compact.replace(/^output:/, '') : null;
  return {
    ...selector,
    metadata: {
      ...selector.metadata,
      ...(anchorSubpath ? { anchor_subpath: anchorSubpath } : {}),
      ...(nodeId && outputPort
        ? {
            graph_port_node_id: nodeId,
            graph_port_name: outputPort,
            graph_port_direction: 'output',
          }
        : {}),
    },
  };
}

function anchorSelector(
  binding: RepresentationBinding,
  entity: StudioScenarioEntity
): StudioSelectorRef | null {
  if (binding?.kind === 'selector') {
    return selectorWithRepresentationMetadata(
      binding.selector,
      binding.anchor_subpath ?? null,
      entity
    );
  }
  if (binding?.kind === 'trial_spec_path') {
    return {
      namespace: 'task_data',
      compact: `task_data:${binding.path}`,
      target_id: entity.scenario_id ?? null,
      path: binding.path,
      role: binding.path.startsWith('targets.') ? 'observed' : 'editable',
      expected_shape: binding.dim ? ['time', binding.dim] : null,
      dtype: null,
      units: null,
      frame: null,
      metadata: {
        label: entity.label,
        source: 'representation_trial_spec_path',
        trial_spec_path: binding.path,
        ...(binding.dim ? { dim: binding.dim } : {}),
      },
    };
  }
  return null;
}

function anchorPositionsForElement(
  component: ComponentDefinition,
  nodeOrTask: ComponentSpec | { params?: Record<string, unknown> | null },
  element: RepresentationElementSpec,
  existingAnchors: Map<string, [number, number] | null>
): Array<[number, number]> {
  return (element.anchors ?? [])
    .map((anchorId) => existingAnchors.get(anchorId) ?? null)
    .filter((point): point is [number, number] => point !== null);
}

function resolveComponentRepresentation({
  scene,
  registry,
  graph,
  component,
  nodeOrTask,
  representation,
  entity,
  nodeId,
}: {
  scene: ResolvedScene;
  registry: StudioScenarioEntityRegistry;
  graph: GraphSpec | null | undefined;
  component: ComponentDefinition;
  nodeOrTask: ComponentSpec | { params?: Record<string, unknown> | null };
  representation: RepresentationSpec;
  entity: StudioScenarioEntity;
  nodeId: string | null;
}) {
  const baseFrame = representation.frame ?? 'world.xy';
  const frame = frameFromProvider(
    representation.frame_provider,
    baseFrame,
    graph,
    nodeId,
    scene.validation,
    entity.id,
    'representation.frame_provider'
  );
  const entityAnchorIds: string[] = [];
  const entityElementIds: string[] = [];
  const localAnchorPositions = new Map<string, [number, number] | null>();
  const planarChain = representation.elements?.find(
    (element) => element.archetype === 'planar_chain'
  );
  const linkLengthPath =
    planarChain?.bindings?.link_lengths?.kind === 'param_path'
      ? planarChain.bindings.link_lengths.path
      : null;
  const twoLinkPoints =
    planarChain?.metadata?.chain_kind === 'two_link_arm' || component.name === 'TwoLinkArm'
      ? twoLinkRestPoints(component, nodeOrTask as ComponentSpec, linkLengthPath)
      : null;

  for (const anchor of representation.anchors ?? []) {
    collectRequiredSelectors(scene.required_selectors, anchor.binding);
    let position = numericPair(bindingValue(component, nodeOrTask, anchor.binding));
    if (!position && twoLinkPoints) {
      const jointIndex = Number(anchor.metadata?.joint_index);
      if (Number.isInteger(jointIndex) && twoLinkPoints[jointIndex]) {
        position = twoLinkPoints[jointIndex];
      } else if (anchor.semantic_role === 'endpoint') {
        position = twoLinkPoints[twoLinkPoints.length - 1];
      }
    }
    if (!position && nodeId === null) {
      position = taskAuthoringAnchorPosition(component, nodeOrTask, anchor);
    }
    localAnchorPositions.set(anchor.id, position);
    const id = anchorGlobalId(entity.id, anchor.id);
    entityAnchorIds.push(id);
    const selector = anchorSelector(anchor.binding, entity);
    const interactionRoles = interactionRoleSet(anchor.interaction_roles);
    const anchorMetadata = {
      ...(representation.metadata?.temporality
        ? { temporality: representation.metadata.temporality }
        : {}),
      ...(representation.metadata?.canonical_goal_anchor
        ? { canonical_goal_anchor: representation.metadata.canonical_goal_anchor }
        : {}),
      ...(anchor.metadata ?? {}),
    };
    scene.anchors.push({
      id,
      entity_id: entity.id,
      local_id: anchor.id,
      label: anchor.label ?? anchor.id,
      semantic_role: anchor.semantic_role,
      position,
      selectable: interactionRoles.has('selectable'),
      hoverable: interactionRoles.has('hoverable'),
      interaction_roles: [...interactionRoles],
      objective_roles: objectiveRolesForAnchor({ ...anchor, metadata: anchorMetadata }, selector),
      frame: anchor.frame ?? frame,
      selector,
      metadata: anchorMetadata,
    });
  }

  for (const element of representation.elements ?? []) {
    for (const binding of Object.values(element.bindings ?? {})) {
      collectRequiredSelectors(scene.required_selectors, binding);
    }
    for (const style of element.style ?? []) {
      collectRequiredSelectors(scene.required_selectors, style.binding);
    }
    const elementFrame = frameFromProvider(
      element.frame_provider,
      element.frame ?? frame,
      graph,
      nodeId,
      scene.validation,
      entity.id,
      `elements.${element.id}.frame_provider`
    );
    let geometry: ResolvedSceneGeometry = { kind: 'none' };
    if (element.archetype === 'planar_chain') {
      const points = twoLinkPoints ?? anchorPositionsForElement(component, nodeOrTask, element, localAnchorPositions);
      geometry = points.length >= 2 ? { kind: 'polyline', points } : { kind: 'none' };
    } else if (element.archetype === 'region') {
      const binding = element.bindings?.bounds;
      const bounds =
        binding?.kind === 'param_path'
          ? boundsParam(component, nodeOrTask, binding.path)
          : null;
      geometry = bounds ? { kind: 'bounds', min: bounds.min, max: bounds.max } : { kind: 'none' };
    } else if (element.archetype === 'marker') {
      const points = anchorPositionsForElement(component, nodeOrTask, element, localAnchorPositions);
      geometry = points.length > 0 ? { kind: 'points', points } : { kind: 'none' };
    } else if (element.archetype === 'distribution_glyph') {
      geometry = reachDistributionGeometry(component, nodeOrTask, element);
    } else if (element.archetype === 'objective_link') {
      continue;
    }

    const id = elementGlobalId(entity.id, element.id);
    entityElementIds.push(id);
    scene.elements.push({
      id,
      entity_id: entity.id,
      local_id: element.id,
      archetype: element.archetype,
      anchor_ids: (element.anchors ?? []).map((anchorId) => anchorGlobalId(entity.id, anchorId)),
      frame: elementFrame,
      scale_invariant: element.scale_invariant ?? representation.scale_invariant ?? false,
      style: styleMap(element.style),
      geometry,
      metadata: element.metadata ?? {},
    });
  }

  scene.entities.push({
    id: entity.id,
    label: entity.label,
    kind: entity.kind,
    summary: entity.summary ?? null,
    related_entity_ids: entityRelations(entity),
    anchor_ids: entityAnchorIds,
    element_ids: entityElementIds,
  });
}

function reachDistributionGeometry(
  component: ComponentDefinition,
  task: { params?: Record<string, unknown> | null },
  element: RepresentationElementSpec
): ResolvedSceneGeometry {
  const workspaceBinding = element.bindings?.workspace;
  const bounds =
    workspaceBinding?.kind === 'param_path'
      ? boundsParam(component, task, workspaceBinding.path)
      : null;
  const center: [number, number] = bounds
    ? [(bounds.min[0] + bounds.max[0]) / 2, (bounds.min[1] + bounds.max[1]) / 2]
    : [0, 0];
  const radius = numberParamValue(component, task, 'eval_reach_length', 0.5);
  const directions = Math.max(1, Math.min(48, Math.round(numberParamValue(component, task, 'eval_n_directions', 7))));
  const points = Array.from({ length: directions }, (_, index): [number, number] => {
    const theta = (Math.PI * 2 * index) / directions;
    return [center[0] + Math.cos(theta) * radius, center[1] + Math.sin(theta) * radius];
  });
  return { kind: 'points', points };
}

function addPlaceholderEntity(
  scene: ResolvedScene,
  entity: StudioScenarioEntity,
  message: string
) {
  const anchorId = anchorGlobalId(entity.id, 'placeholder');
  const elementId = elementGlobalId(entity.id, 'placeholder');
  scene.validation.push({
    type: 'workspace_unrepresented_entity',
    severity: 'warning',
    message,
    entity_id: entity.id,
  });
  scene.anchors.push({
    id: anchorId,
    entity_id: entity.id,
    local_id: 'placeholder',
    label: entity.label,
    semantic_role: 'glyph',
    position: [0, 0],
    selectable: true,
    hoverable: true,
    frame: scene.frame,
    selector: entity.selector,
    interaction_roles: ['selectable', 'hoverable'],
    objective_roles: [],
    metadata: {},
  });
  scene.elements.push({
    id: elementId,
    entity_id: entity.id,
    local_id: 'placeholder',
    archetype: 'annotation',
    anchor_ids: [anchorId],
    frame: scene.frame,
    scale_invariant: true,
    style: {},
    geometry: { kind: 'points', points: [[0, 0]] },
    metadata: { placeholder: true },
  });
  scene.entities.push({
    id: entity.id,
    label: entity.label,
    kind: entity.kind,
    summary: entity.summary ?? null,
    related_entity_ids: entityRelations(entity),
    anchor_ids: [anchorId],
    element_ids: [elementId],
  });
}

function isObjectiveSpec(value: StudioScenarioSpec['objective_spec']): value is StudioObjectiveSpec {
  return Boolean(value && typeof value === 'object' && Array.isArray((value as StudioObjectiveSpec).terms));
}

function anchorSelectorMatches(
  anchor: ResolvedSceneAnchor,
  selector: StudioSelectorRef | null | undefined
): boolean {
  if (!selector || !anchor.selector) return false;
  if (anchor.selector.compact === selector.compact) return true;
  if (selectorToEntityId(anchor.selector) && selectorToEntityId(anchor.selector) === selectorToEntityId(selector)) {
    return true;
  }
  const selectorMetadata = selector.metadata ?? {};
  const anchorMetadata = anchor.selector.metadata ?? {};
  return (
    selectorMetadata.graph_port_node_id === anchorMetadata.graph_port_node_id &&
    selectorMetadata.graph_port_name === anchorMetadata.graph_port_name &&
    selectorMetadata.graph_port_name === anchor.local_id
  );
}

function anchorForObjectiveSelector(
  scene: ResolvedScene,
  selector: StudioSelectorRef | null | undefined,
  role: 'source' | 'target'
): ResolvedSceneAnchor | null {
  const candidates = scene.anchors.filter(
    (anchor) =>
      anchor.position &&
      anchorSelectorMatches(anchor, selector) &&
      (role === 'source'
        ? anchor.objective_roles.includes('objective-source')
        : anchor.objective_roles.includes('objective-target'))
  );
  return candidates[0] ?? null;
}

function addObjectiveProjectionEntity(
  scene: ResolvedScene,
  registry: StudioScenarioEntityRegistry,
  term: StudioObjectiveTermSpec
) {
  const entity = registry.entities[objectiveEntityId(term.id)];
  if (!entity) return;
  const sourceAnchor = anchorForObjectiveSelector(scene, term.source_selector, 'source');
  const targetAnchor = anchorForObjectiveSelector(scene, term.target_selector, 'target');
  if (!sourceAnchor?.position || !targetAnchor?.position) return;
  const elementId = `${entity.id}::element:objective_link`;
  scene.elements.push({
    id: elementId,
    entity_id: entity.id,
    local_id: 'objective_link',
    archetype: 'objective_link',
    anchor_ids: [sourceAnchor.id, targetAnchor.id],
    frame: sourceAnchor.frame,
    scale_invariant: true,
    style: {},
    geometry: { kind: 'link', points: [sourceAnchor.position, targetAnchor.position] },
    metadata: {
      term_id: term.id,
      source_anchor_id: sourceAnchor.id,
      target_anchor_id: targetAnchor.id,
      timing: term.temporal_selector ?? term.metadata.target_timing ?? null,
    },
  });
  scene.entities.push({
    id: entity.id,
    label: entity.label,
    kind: entity.kind,
    summary: entity.summary ?? null,
    related_entity_ids: entityRelations(entity),
    anchor_ids: [sourceAnchor.id, targetAnchor.id],
    element_ids: [elementId],
  });
}

function addObjectiveProjections(
  scene: ResolvedScene,
  registry: StudioScenarioEntityRegistry,
  objectiveSpec: StudioObjectiveSpec | null
) {
  for (const term of objectiveSpec?.terms ?? []) {
    addObjectiveProjectionEntity(scene, registry, term);
  }
}

function reachabilityMessages(
  scene: ResolvedScene,
  registry: StudioScenarioEntityRegistry,
  graph: GraphSpec | null | undefined,
  componentsByName: Map<string, ComponentDefinition>,
  task: StudioScenarioSpec['task_spec']
) {
  const armEntry = Object.entries(graph?.nodes ?? {}).find(
    ([, node]) => node.type === 'TwoLinkArm'
  );
  const armComponent = componentsByName.get('TwoLinkArm');
  const taskComponent = task ? componentForTask(task.type, componentsByName) : null;
  if (!armEntry || !armComponent || !taskComponent || !task) return;
  const [nodeId, armNode] = armEntry;
  const lengths = numericArray(paramValue(armComponent, armNode, 'link_lengths')) ?? [0.3, 0.33];
  const reach = lengths.reduce((total, length) => total + Math.abs(length), 0);
  const taskEntity = registry.entities[taskEntityId(registry.scenario_id)];
  const radius = numberParamValue(taskComponent, task, 'eval_reach_length', 0.5);
  const workspace = boundsParam(taskComponent, task, 'workspace');
  if (radius > reach) {
    scene.validation.push({
      type: 'workspace_goal_out_of_reach',
      severity: 'warning',
      message: `Reach target radius ${radius.toFixed(3)} m exceeds TwoLinkArm reach ${reach.toFixed(3)} m.`,
      entity_id: taskEntity?.id ?? null,
      path: 'task_spec.params.eval_reach_length',
    });
  }
  if (workspace) {
    const corners: Array<[number, number]> = [
      workspace.min,
      [workspace.min[0], workspace.max[1]],
      [workspace.max[0], workspace.min[1]],
      workspace.max,
    ];
    const farthest = Math.max(...corners.map(([x, y]) => Math.hypot(x, y)));
    if (farthest > reach) {
      scene.validation.push({
        type: 'workspace_region_partially_out_of_reach',
        severity: 'warning',
        message: `Workspace bounds extend to ${farthest.toFixed(3)} m from the shoulder, beyond TwoLinkArm reach ${reach.toFixed(3)} m.`,
        entity_id: taskEntity?.id ?? mechanicsEntityId(registry.scenario_id, nodeId),
        path: 'task_spec.params.workspace',
      });
    }
  }
}

export function buildResolvedScene({
  scenario,
  graph,
  registry,
  components,
}: {
  scenario: StudioScenarioSpec | null | undefined;
  graph?: GraphSpec | null;
  registry: StudioScenarioEntityRegistry;
  components: ComponentDefinition[];
}): ResolvedScene {
  const resolvedGraph = graph ?? scenario?.graph ?? null;
  const componentsByName = componentByName(components);
  const scene: ResolvedScene = {
    scenario_id: scenario?.id ?? registry.scenario_id,
    frame: 'world.xy',
    units: 'm',
    entities: [],
    anchors: [],
    elements: [],
    required_selectors: [],
    validation: [],
  };

  for (const [nodeId, node] of Object.entries(resolvedGraph?.nodes ?? {})) {
    const component = componentsByName.get(node.type);
    const representation = component?.representation ?? null;
    if (!component || !representation) {
      const entity = registry.entities[mechanicsEntityId(scenario?.id ?? registry.scenario_id, nodeId)];
      if (entity) {
        addPlaceholderEntity(
          scene,
          entity,
          `Node '${nodeId}' has no workspace representation metadata.`
        );
      }
      continue;
    }
    const mechanicsEntity = registry.entities[mechanicsEntityId(scenario?.id ?? registry.scenario_id, nodeId)];
    if (!mechanicsEntity) continue;
    resolveComponentRepresentation({
      scene,
      registry,
      graph: resolvedGraph,
      component,
      nodeOrTask: node,
      representation,
      entity: mechanicsEntity,
      nodeId,
    });
  }

  if (scenario?.task_spec) {
    const taskEntity = registry.entities[taskEntityId(scenario.id)];
    const component = componentForTask(scenario.task_spec.type, componentsByName);
    if (taskEntity && component?.representation) {
      resolveComponentRepresentation({
        scene,
        registry,
        graph: resolvedGraph,
        component,
        nodeOrTask: scenario.task_spec,
        representation: component.representation,
        entity: taskEntity,
        nodeId: null,
      });
    } else if (taskEntity) {
      addPlaceholderEntity(
        scene,
        taskEntity,
        `Task '${scenario.task_spec.type}' has no workspace representation metadata.`
      );
    }
  }

  reachabilityMessages(
    scene,
    registry,
    resolvedGraph,
    componentsByName,
    scenario?.task_spec ?? null
  );
  addObjectiveProjections(
    scene,
    registry,
    isObjectiveSpec(scenario?.objective_spec) ? scenario.objective_spec : null
  );

  return scene;
}
