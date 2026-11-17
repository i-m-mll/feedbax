import { bindStudioCommand } from '@/features/projectWindow/studioCommands';
import { appearanceSettings, comparisonSettings } from './appearance';

export function appearanceCommands() {
  return [...appearanceSettings(), ...comparisonSettings()].map(({ id, apply }) => bindStudioCommand(id, () => apply(window)));
}
