import { plainText } from '@/components/ui/ShownText';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import { rawField } from '@/components/values/adapters/raw';
import type { FieldModel } from '@/components/values/fieldModel';
import type { InterventionAttachment, JsonValue, TimeValue, Window } from '@/generated/studioContracts';
import { windowText, type OffsetSite, type WindowEpochChoice, type WindowHost } from '../WindowEditor';

function offsetAt(window: Window | null | undefined, site: OffsetSite): JsonValue | undefined {
  let node: unknown = window;
  for (const key of site) node = (node as Record<string | number, unknown> | undefined)?.[key];
  return node as JsonValue | undefined;
}

function withOffset(window: Window, site: OffsetSite, offset: TimeValue): Window {
  const [, index, at] = site;
  return { spans: window.spans.map((span, position) => position !== index ? span
    : { ...span, [at]: { ...(span as unknown as Record<string, object>)[at], offset } } as Window['spans'][number]) };
}

/** A duration that may be negative or zero, in seconds or whole steps. */
function signedDuration(raw: string): TimeValue | undefined {
  const match = raw.trim().match(/^([-+]?(?:\d+(?:\.\d+)?|\.\d+))\s*(s|st)$/);
  if (!match) return undefined;
  const value = Number(match[1]);
  const unit = match[2] as TimeValue['unit'];
  return unit === 's' || Number.isInteger(value) ? { value, unit } : undefined;
}

/** An offset the intervention stores as written: a duration in either unit, or the Task variable it reads. */
function rawOffset(item: InterventionAttachment, site: OffsetSite, linked: (reference: string) => string,
  change?: (item: InterventionAttachment) => unknown): FieldModel {
  const label = studioName(site[2]);
  const offset = offsetAt(item.window, site);
  const reference = offset && typeof offset === 'object' && !Array.isArray(offset) && offset.kind === 'name' ? String(offset.reference) : undefined;
  const id = site.join('-');
  if (reference !== undefined) return rawField({ id, label, type: { kind: 'readOnly' }, value: linked(reference) });
  const time = offset as TimeValue | undefined;
  return rawField({
    id, label, type: { kind: 'string' }, value: time ? `${time.value} ${time.unit}` : '',
    change: change && item.window ? ((raw) => {
      const next = signedDuration(String(raw));
      if (!next) return studioReason('takesDuration', label);
      void change({ ...item, window: withOffset(item.window!, site, next) });
    }) : undefined,
  });
}

/**
 * An intervention's Window as the editor hosts it: each offset the field the Service projects for it (`projected`), or
 * else the offset as stored; its structure stored on the intervention, none being the whole trial.
 */
export function interventionWindow({ item, epochs, projected, linked, change }: {
  item: InterventionAttachment;
  epochs: readonly WindowEpochChoice[];
  projected: (site: OffsetSite) => FieldModel | undefined;
  linked: (reference: string) => string;
  change?: (item: InterventionAttachment) => unknown;
}): WindowHost {
  return {
    epochs,
    offset: (site) => projected(site) ?? rawOffset(item, site, linked, change),
    change: change && ((window) => {
      const { window: _whole, ...rest } = item;
      return change(window ? { ...item, window } : rest);
    }),
  };
}

/** The text of an intervention's Window, the whole trial when it has none. */
export function windowDescription(item: InterventionAttachment, epochs: readonly WindowEpochChoice[], linked: (reference: string) => string) {
  return plainText(windowText(item.window, interventionWindow({ item, epochs, projected: () => undefined, linked }))) || studioName('wholeTrial');
}
