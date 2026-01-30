export type ParamPrimitive = number | string | boolean | null;

export interface ParamValueObject {
  [key: string]: ParamValue;
}

export interface ParamValueArray extends Array<ParamValue> {}

export type ParamValue = ParamPrimitive | ParamValueArray | ParamValueObject;

export interface ParamSchema {
  name: string;
  type: 'int' | 'float' | 'bool' | 'str' | 'enum' | 'array' | 'object';
  default?: ParamValue;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  description?: string;
  required: boolean;
  nested_schema?: ParamSchema[];
}

export interface ComponentSpec {
  type: string;
  params: Record<string, ParamValue>;
  input_ports: string[];
  output_ports: string[];
}

export interface WireSpec {
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
}

export interface GraphSpec {
  nodes: Record<string, ComponentSpec>;
  wires: WireSpec[];
  input_ports: string[];
  output_ports: string[];
  input_bindings: Record<string, [string, string]>;
  output_bindings: Record<string, [string, string]>;
  subgraphs?: Record<string, GraphSpec>;
  metadata?: GraphMetadata;
}

export interface GraphMetadata {
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  version: string;
  author?: string;
  tags?: string[];
}

export interface NodeUIState {
  position: { x: number; y: number };
  collapsed: boolean;
  selected: boolean;
  size?: { width: number; height: number };
}

export interface EdgeRoutingPoint {
  x: number;
  y: number;
}

export interface EdgeRouting {
  style: 'bezier' | 'elbow';
  points: EdgeRoutingPoint[];
}

export interface EdgeUIState {
  routing: EdgeRouting;
}

export interface GraphUIState {
  viewport: { x: number; y: number; zoom: number };
  node_states: Record<string, NodeUIState>;
  edge_states?: Record<string, EdgeUIState>;
  subgraph_states?: Record<string, GraphUIState>;
}

export interface GraphNodeData extends Record<string, unknown> {
  label: string;
  spec: ComponentSpec;
  collapsed?: boolean;
  size?: { width: number; height: number };
}

export interface GraphEdgeData extends Record<string, unknown> {
  routing?: EdgeRouting;
}
