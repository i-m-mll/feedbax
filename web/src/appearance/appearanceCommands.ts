import { bindStudioCommand } from '@/features/projectWindow/studioCommands';
import { appearanceSettings } from './appearance';

export function appearanceCommands() {
  return appearanceSettings().map(({ id, apply }) => bindStudioCommand(id, () => apply(window)));
}
