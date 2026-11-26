import { plainText } from '@/components/ui/ShownText';
import { studioName, studioReason } from '@/components/activity/studioVocabulary';
import { rawField } from '@/components/values/adapters/raw';
import type { FieldModel } from '@/components/values/fieldModel';
import type { StoredExpression, StoredWindow } from '@/generated/studioContracts';
import { windowText, type OffsetSite, type WindowEpochChoice, type WindowHost } from '../WindowEditor';
import type { Attachment } from './modelGraph';

type Intervention = Extract<Attachment, { kind: 'intervention' }>;

function offsetAt(window: StoredWindow | null | undefined, site: OffsetSite): StoredExpression | null | undefined {
  let node: unknown = window;
  for (const key of site) node = (node as Record<string | number, unknown> | undefined)?.[key];
  return node as StoredExpression | null | undefined;
}

function withOffset(window: StoredWindow, site: OffsetSite, offset: StoredExpression): StoredWindow {
  const [, index, at] = site;
  return { spans: window.spans.map((span, position) => position !== index ? span
    : { ...span, [at]: { ...(span as unknown as Record<string, object>)[at], offset } } as StoredWindow['spans'][number]) };
}

/** An offset as the intervention stores it, edited as the text the Service reads. */
function rawOffset(item: Intervention, site: OffsetSite, change?: (item: Intervention) => unknown): FieldModel {
  const label = studioName(site[2]);
  const offset = offsetAt(item.window, site);
  return rawField({
    id: site.join('-'), label, type: { kind: 'string' }, value: offset == null ? '' : String(offset),
    change: change && item.window ? ((raw) => {
      const text = String(raw).trim();
      if (!text) return studioReason('takesDuration', label);
      void change({ ...item, window: withOffset(item.window!, site, text) });
    }) : undefined,
  });
}

/**
 * An intervention's Window as the editor hosts it: each offset the field the Service projects for it (`projected`), or
 * else the offset as stored; its structure stored on the intervention, none being the whole trial.
 */
export function interventionWindow({ item, epochs, projected, change }: {
  item: Intervention;
  epochs: readonly WindowEpochChoice[];
  projected: (site: OffsetSite) => FieldModel | undefined;
  change?: (item: Intervention) => unknown;
}): WindowHost {
  return {
    epochs,
    offset: (site) => projected(site) ?? rawOffset(item, site, change),
    change: change && ((window) => {
      const { window: _whole, ...rest } = item;
      return change(window ? { ...item, window } : rest);
    }),
  };
}

/** The text of an intervention's Window, the whole trial when it has none. */
export function windowDescription(item: Intervention, epochs: readonly WindowEpochChoice[]) {
  return plainText(windowText(item.window, interventionWindow({ item, epochs, projected: () => undefined }))) || studioName('wholeTrial');
}
