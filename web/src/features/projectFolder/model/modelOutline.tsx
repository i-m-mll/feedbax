import { Box, Boxes, CircleDot, Paperclip, Spline } from 'lucide-react';
import type { ReactNode } from 'react';
import type { TreeNode } from '@/components/ui/Tree';
import { studioName } from '@/components/activity/studioVocabulary';
import type { ModelAddress, ModelDocument, ModelPortAttachments, ModelWire } from '@/generated/studioContracts';
import type { Attachment } from './ModelInspector';
import { addressKey, at, attachmentAddress, boundaryPorts, componentPorts, modelPath, type Component, type GraphOwner, type Port, type Registry } from './modelGraph';

export interface ModelOutline {
  nodes: TreeNode[];
  /** Each row's element, by row id. */
  addresses: ReadonlyMap<string, ModelAddress>;
  /** The rows with rows inside them. */
  parents: ReadonlySet<string>;
}

export interface OutlineInput {
  document: string;
  model: ModelDocument;
  /** The model with its attachments as they apply here, when that differs from `model`. */
  effective?: ModelDocument;
  registry: Registry;
  attachmentName: (item: Attachment, address: ModelAddress) => string;
  trailing: (address: ModelAddress) => ReactNode;
}

/** The Model's elements as a tree: a composite holds its interior, a wire or port holds its attachments. */
function endName(owner: GraphOwner, end: ModelWire['source'] | ModelWire['target'], registry: Registry) {
  if (end.owner !== 'component') {
    const port = boundaryPorts(owner.ports).find((item) => item.end.owner === end.owner && item.end.port === end.port);
    return `${studioName(end.owner === 'outputs' ? 'outputs' : 'inputs')}.${port?.name ?? end.port}`;
  }
  const component = owner.interior.components?.[end.component];
  const port = component && componentPorts(end.component, component, registry).find((item) => item.end.port === end.port);
  return `${component?.name ?? end.component}.${port?.name ?? end.port}`;
}

export function modelOutline({ document, model, effective, registry, attachmentName, trailing }: OutlineInput): ModelOutline {
  const addresses = new Map<string, ModelAddress>();
  const parents = new Set<string>();
  const row = (address: ModelAddress, name: string, glyph: ReactNode, children: TreeNode[] = [], renamable = false): TreeNode => {
    const id = addressKey(address);
    addresses.set(id, address);
    if (children.length) parents.add(id);
    return { id, name, glyph, children: children.length ? children : undefined, renamable, trailing: trailing(address) };
  };
  const attached = (owner: ModelAddress, list: readonly Attachment[] | undefined) => (list ?? []).map((item) => {
    const address = attachmentAddress(owner, item.id);
    return row(address, attachmentName(item, address), <Paperclip />);
  });
  const portRows = (graph: ModelAddress, ports: readonly Port[], store: ModelPortAttachments | undefined) =>
    ports.flatMap((port) => {
      if (port.direction === 'physical') return [];
      const list = (port.direction === 'output' ? store?.sources : store?.sinks)?.[port.end.port];
      if (!list?.length) return [];
      const address = at(graph, { kind: 'port', end: port.end });
      return [row(address, port.name, <CircleDot />, attached(address, list))];
    });

  const graphRows = (path: readonly string[]): TreeNode[] => {
    const owner = modelPath(model, path).owner;
    const applied = effective ? modelPath(effective, path).owner.interior : owner.interior;
    const graph: ModelAddress = { document, composites: [...path], element: { kind: 'graph' } };
    const components = Object.entries(owner.interior.components ?? {}).map(([id, component]: [string, Component]) => {
      const address = at(graph, { kind: 'component', id });
      const ports = portRows(graph, componentPorts(id, component, registry), applied.components?.[id]?.attachments);
      const inside = component.kind === 'composite' ? graphRows([...path, id]) : [];
      return row(address, component.name, component.kind === 'composite' ? <Boxes /> : <Box />, [...ports, ...inside], true);
    });
    const wires = Object.entries(owner.interior.wires ?? {}).map(([id, wire]) => {
      const address = at(graph, { kind: 'wire', id });
      return row(address, `${endName(owner, wire.source, registry)} → ${endName(owner, wire.target, registry)}`, <Spline />, attached(address, applied.wires?.[id]?.attachments));
    });
    return [...portRows(graph, boundaryPorts(owner.ports), applied.attachments), ...components, ...wires];
  };

  return { nodes: graphRows([]), addresses, parents };
}
