import type { ComponentDefinition } from '@/types/components';

/**
 * Offline component registry fallback.
 *
 * The backend registry is the source of truth. Keep this fallback empty so an
 * unreachable backend is surfaced as an explicit empty/error state instead of a
 * stale hand-maintained registry.
 */
export const componentLibrary: ComponentDefinition[] = [];
