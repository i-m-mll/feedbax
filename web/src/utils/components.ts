import type { ComponentDefinition } from '@/types/components';

export function groupComponentsByCategory(components: ComponentDefinition[]) {
  return components.reduce<Record<string, ComponentDefinition[]>>((acc, component) => {
    acc[component.category] = acc[component.category] || [];
    acc[component.category].push(component);
    return acc;
  }, {});
}
