import type { ComponentSpec, ParamSchema } from '@/types/graph';

export interface ComponentDefinition {
  name: string;
  category: string;
  description: string;
  param_schema: ParamSchema[];
  input_ports: string[];
  output_ports: string[];
  icon: string;
  default_params: ComponentSpec['params'];
}
