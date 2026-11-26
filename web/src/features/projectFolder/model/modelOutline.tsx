import { Box, Boxes, CircleDot, Paperclip, Spline } from 'lucide-react';
import type { ReactNode } from 'react';
import type { TreeNode } from '@/components/ui/Tree';
import { joinText, referentName, type ShownText } from '@/components/ui/ShownText';
import { studioName } from '@/components/activity/studioVocabulary';
import type { ModelAddress, StoredModelDocument, StoredModelPortAttachments } from '@/generated/studioContracts';
import {
  addressKey, at, attachmentAddress, boundaryPorts, chainItems, componentPorts, modelPath,
  type Attachment, type AttachmentChain, type Component, type GraphOwner, type ModelWire, type Port, type Registry,
} from './modelGraph';

export interface ModelOutline {
  nodes: TreeNode[];
  /** Each row's element, by row id. */
  addresses: ReadonlyMap<string, ModelAddress>;
  /** The rows with rows inside them. */
  parents: ReadonlySet<string>;
}

export interface OutlineInput {
  model: StoredModelDocument;
  /** The model with its attachments as they apply here, when that differs from `model`. */
  effective?: StoredModelDocument;
  registry: Registry;
  attachmentName: (item: Attachment, address: ModelAddress) => string;
  trailing: (address: ModelAddress) => ReactNode;
  /** Whether the element is drawn only because it was removed: its row is struck through. */
  removed: (address: ModelAddress) => boolean;
}

/** The Model's elements as a tree: a composite holds its interior, a wire or port holds its attachments. */
function endName(owner: GraphOwner, end: ModelWire['source'] | ModelWire['target'], registry: Registry): ShownText {
  if (end.owner !== 'component') {
    const port = boundaryPorts(owner.ports).find((item) => item.end.owner === end.owner && item.end.port === end.port);
    return joinText([studioName(end.owner === 'outputs' ? 'outputs' : 'inputs'), [referentName(port, end.port)]], '.');
  }
  const component = owner.interior.components?.[end.component];
  const port = component && componentPorts(end.component, component, registry).find((item) => item.end.port === end.port);
  return joinText([[component ? component.label : { missing: end.component }], [referentName(port, end.port)]], '.');
}

export function modelOutline({ model, effective, registry, attachmentName, trailing, removed }: OutlineInput): ModelOutline {
  const addresses = new Map<string, ModelAddress>();
  const parents = new Set<string>();
  const row = (address: ModelAddress, name: ShownText, glyph: ReactNode, children: TreeNode[] = [], renamable = false): TreeNode => {
    const id = addressKey(address);
    addresses.set(id, address);
    if (children.length) parents.add(id);
    const struck = removed(address);
    return { id, name, glyph, children: children.length ? children : undefined, renamable: renamable && !struck, struck, trailing: trailing(address) };
  };
  const attached = (owner: ModelAddress, chain: AttachmentChain | undefined) => chainItems(chain).map((item) => {
    const address = attachmentAddress(owner, item.id);
    return row(address, attachmentName(item, address), <Paperclip />);
  });
  const portRows = (graph: ModelAddress, ports: readonly Port[], store: StoredModelPortAttachments | undefined) =>
    ports.flatMap((port) => {
      if (port.direction === 'physical') return [];
      const list = (port.direction === 'output' ? store?.sources : store?.sinks)?.[port.end.port];
      if (!list || !Object.keys(list).length) return [];
      const address = at(graph, { kind: 'port', end: port.end });
      return [row(address, port.name, <CircleDot />, attached(address, list))];
    });

  const graphRows = (path: readonly string[]): TreeNode[] => {
    const owner = modelPath(model, path).owner;
    const applied = effective ? modelPath(effective, path).owner.interior : owner.interior;
    const graph: ModelAddress = { composites: [...path], element: { kind: 'graph' } };
    const components = Object.entries(owner.interior.components ?? {}).map(([id, component]: [string, Component]) => {
      const address = at(graph, { kind: 'component', id });
      const ports = portRows(graph, componentPorts(id, component, registry), applied.components?.[id]?.attachments);
      const inside = component.kind === 'composite' ? graphRows([...path, id]) : [];
      return row(address, component.label, component.kind === 'composite' ? <Boxes /> : <Box />, [...ports, ...inside], true);
    });
    const wires = Object.entries(owner.interior.wires ?? {}).map(([id, wire]) => {
      const address = at(graph, { kind: 'wire', id });
      return row(address, joinText([endName(owner, wire.source, registry), endName(owner, wire.target, registry)], ' → '), <Spline />, attached(address, applied.wires?.[id]?.attachments));
    });
    return [...portRows(graph, boundaryPorts(owner.ports), applied.attachments), ...components, ...wires];
  };

  return { nodes: graphRows([]), addresses, parents };
}
