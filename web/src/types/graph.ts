export type ParamPrimitive = number | string | boolean | null;

export interface ParamValueObject {
  [key: string]: ParamValue;
}

export interface ParamValueArray extends Array<ParamValue> {}

export type ParamValue = ParamPrimitive | ParamValueArray | ParamValueObject;

export interface ParamSchema {
  name: string;
  type: 'int' | 'float' | 'bool' | 'str' | 'enum' | 'array' | 'object' | 'bounds2d';
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

export interface UserPortSpec {
  inputs: string[];
  outputs: string[];
}

export interface TapTransform {
  type: string;
  params: ParamValueObject;
}

export interface TapSpec {
  id: string;
  type: 'probe' | 'intervention';
  position: {
    afterNode: string;
    targetNode?: string;
  };
  paths: Record<string, string>;
  transform?: TapTransform;
}

export interface TapUIState {
  position: { x: number; y: number };
  selected?: boolean;
}

export type BarnacleKind = 'probe' | 'intervention';
export type BarnacleTiming = 'input' | 'output';

export interface BarnacleSpec {
  id: string;
  kind: BarnacleKind;
  timing: BarnacleTiming;
  label: string;
  read_paths: string[];
  write_paths: string[];
  transform: string;
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
  barnacles?: Record<string, BarnacleSpec[]>;
  user_ports?: Record<string, UserPortSpec>;
  taps?: TapSpec[];
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
  tap_states?: Record<string, TapUIState>;
}

export interface GraphNodeData extends Record<string, unknown> {
  label: string;
  spec: ComponentSpec;
  collapsed?: boolean;
  size?: { width: number; height: number };
}

export interface TapNodeData extends Record<string, unknown> {
  tap: TapSpec;
}

export interface GraphEdgeData extends Record<string, unknown> {
  routing?: EdgeRouting;
}
