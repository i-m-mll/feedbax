import type { ComponentSpec, GraphSpec, GraphUIState, ParamSchema } from '@/types/graph';

export interface PortType {
  dtype: string;
  shape?: number[] | null;
  rank?: number;
}

export interface PortTypeSpec {
  inputs: Record<string, PortType>;
  outputs: Record<string, PortType>;
}

export interface ComponentDefinition {
  name: string;
  category: string;
  description: string;
  param_schema: ParamSchema[];
  input_ports: string[];
  output_ports: string[];
  icon: string;
  default_params: ComponentSpec['params'];
  port_types?: PortTypeSpec;
  is_composite?: boolean;
  /** Pre-filled internal graph for composite components dragged from the palette. */
  template_graph?: GraphSpec;
  /** Default UI positions for nodes in template_graph. */
  template_ui_state?: GraphUIState;
}
