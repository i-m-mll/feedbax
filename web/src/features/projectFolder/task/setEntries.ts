import type { StoredVariable, StudioSetVariableOrigin, StudioTrialAuthoring } from '@/generated/studioContracts';
import { familyText } from '../FamilyOver';
import type { EntryName } from './taskModel';

/** The Set entry an origin addresses, as the origin names it. */
export const entryName = ({ label, member_count }: StudioSetVariableOrigin): EntryName =>
  ({ label, ...member_count != null && { members: member_count } });

/** A Task variable's name and, for one a Set entry adds, that entry: the Service's read names it by its own label. */
export function taskVariableName(view: Pick<StudioTrialAuthoring, 'variables'>, key: string, variable: StoredVariable) {
  const scoped = view.variables.find(({ id }) => id === key);
  return scoped?.set_entry ? { name: scoped.name, entry: entryName(scoped.set_entry) } : { name: variable.label };
}

/** The entry each Set variable of the view is marked with, as text, by the variable's id. */
export function entryMarks(view: Pick<StudioTrialAuthoring, 'variables'>): ReadonlyMap<string, string> {
  return new Map(view.variables.flatMap(({ id, set_entry }) => {
    if (!set_entry) return [];
    const { label, members } = entryName(set_entry);
    return [[id, familyText(label, members)] as const];
  }));
}
