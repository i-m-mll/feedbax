import { useQuery } from '@tanstack/react-query';
import type { NodePath } from '@/generated/studioContracts';
import type { FieldModel, PartMenu } from './fieldModel';

/** The query that holds the menu at `part` of the served menus named `key`. */
export const partMenuKey = (key: readonly unknown[], part: NodePath) => ['part-menu', ...key, part] as const;

const same = (left: NodePath, right: NodePath) =>
  left.length === right.length && left.every((segment, index) => segment === right[index]);

/**
 * The menu of the node at `part` of the field's value. A listed menu is the view's; a served one is read once it is
 * `open`, and offers nothing it has not read for this view.
 */
export function usePartMenu(model: FieldModel, part: NodePath, open: boolean): PartMenu {
  const menus = model.menus;
  const served = menus?.kind === 'served' ? menus : undefined;
  const query = useQuery({
    queryKey: partMenuKey(served?.key ?? [], part),
    queryFn: ({ signal }) => served!.read(part, signal),
    enabled: !!served && open,
    staleTime: Infinity,
    retry: false,
  });
  if (!menus) return { kind: 'read', site: undefined };
  if (menus.kind === 'listed') return { kind: 'read', site: menus.parts.find(({ path }) => same(path, part)) };
  if (query.data) return { kind: 'read', site: query.data };
  if (query.error) return { kind: 'failed', reason: query.error.message };
  return { kind: 'reading' };
}
