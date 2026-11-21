import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { createStore } from 'zustand/vanilla';
import type { ModelAddress } from '@/generated/studioContracts';
import { addressKey, at, type Port } from '@/features/projectFolder/model/modelGraph';

/**
 * An element of a document that has no element grammar of its own yet: an Analysis operation or figure, an Objective
 * term, a Task signal. `field` is the field of it whose cell is selected.
 */
export interface NamedElementAddress {
  readonly document: string;
  readonly element: { readonly kind: 'named'; readonly id: string; readonly field?: string };
}

/** A Model element in the document that holds it; a Model address is relative to its document. */
export interface ModelLocation {
  readonly document: string;
  readonly address: ModelAddress;
}

/** The Model elements one edit changed, in the document it changed. */
export interface ModelChange {
  readonly document: string;
  readonly addresses: readonly ModelAddress[];
}

/** What the window's selection holds: an element of a document. */
export type ElementAddress = ModelLocation | NamedElementAddress;

/** What a click can fill a reference with while picking: a Model port, a step's card, or a named element. */
export type PickTarget =
  | { readonly kind: 'port'; readonly document: string; readonly address: ModelAddress; readonly port: Port }
  | { readonly kind: 'step'; readonly step: string }
  | { readonly kind: 'element'; readonly address: NamedElementAddress };

export interface ReferencePick { accepts(target: PickTarget): boolean; choose(target: PickTarget): void }
type State = {
  selection: readonly ElementAddress[];
  /** A reference field's crosshair: the next element clicked fills it instead of being selected. */
  pick: ReferencePick | null;
  /** A Model element the slot showing its document moves to. */
  reveal: ModelLocation | null;
};

export type ElementSelectionStore = ReturnType<typeof useElementSelectionStore>;

/** The window's one selection of elements, read by every tree, canvas, table and inspector. */
export function useElementSelectionStore() {
  const [store] = useState(() => createStore<State>(() => ({ selection: [], pick: null, reveal: null })));
  return store;
}

export function isModelAddress(address: ElementAddress): address is ModelLocation {
  return 'address' in address;
}

/** The element a selected address stands for: a field is shown as the element that holds it. */
export function selectedElement(address: ModelAddress): ModelAddress {
  return address.element.kind === 'field' ? at(address, address.element.parent) : address;
}

export function elementKey(address: ElementAddress) {
  return isModelAddress(address) ? JSON.stringify([address.document, addressKey(address.address)])
    : JSON.stringify([address.document, address.element.id, address.element.field ?? null]);
}

function sameGroup(left: ElementAddress, right: ElementAddress) {
  if (left.document !== right.document) return false;
  if (!isModelAddress(left) || !isModelAddress(right)) return isModelAddress(left) === isModelAddress(right);
  return left.address.element.kind === right.address.element.kind &&
    JSON.stringify(left.address.composites) === JSON.stringify(right.address.composites);
}

/** Selects one element, or with `extend` toggles it among the selected elements of its graph and kind. */
export function selectElement(store: ElementSelectionStore, address: ElementAddress, extend = false) {
  const current = store.getState().selection;
  if (!extend || current.some((item) => !sameGroup(item, address))) {
    store.setState({ selection: [address] });
    return;
  }
  const key = elementKey(address);
  const other = current.filter((item) => elementKey(item) !== key);
  store.setState({ selection: other.length === current.length ? [...other, address] : other });
}

export function pickKey(target: PickTarget) {
  switch (target.kind) {
    case 'port': return JSON.stringify(['port', target.document, addressKey(target.address)]);
    case 'step': return JSON.stringify(['step', target.step]);
    case 'element': return JSON.stringify(['element', elementKey(target.address)]);
  }
}

const unpicked = createStore<State>(() => ({ selection: [], pick: null, reveal: null }));

/** The armed pick, or null while nothing is being picked. */
export function usePick(store: ElementSelectionStore | undefined) {
  return useStore(store ?? unpicked, (state) => state.pick);
}

/** Fills the armed reference with `target` and disarms it. */
export function choosePick(store: ElementSelectionStore, target: PickTarget) {
  const { pick } = store.getState();
  if (!pick?.accepts(target)) return false;
  store.setState({ pick: null });
  pick.choose(target);
  return true;
}

export function clearSelection(store: ElementSelectionStore) {
  store.setState({ selection: [] });
}

/** The selected elements of one Model document, fields resolved to the element that holds them. */
export function useModelSelection(store: ElementSelectionStore, document: string | undefined) {
  const selection = useStore(store, (state) => state.selection);
  return useMemo(() => selection.flatMap((item) => isModelAddress(item) && item.document === document
    ? [selectedElement(item.address)] : []), [document, selection]);
}

/**
 * The selected elements of one document without an element grammar, as ids; `selectedId` is the one selected element,
 * when only one is. `reveal`, when it changes, selects `revealedId`.
 */
export function useNamedSelection(store: ElementSelectionStore, document: string | undefined, reveal?: object, revealedId?: string) {
  const selection = useStore(store, (state) => state.selection);
  const handled = useRef<object>();
  useEffect(() => {
    if (!document || !reveal || revealedId === undefined || handled.current === reveal) return;
    handled.current = reveal;
    store.setState({ selection: [{ document, element: { kind: 'named', id: revealedId } }] });
  }, [document, reveal, revealedId, store]);
  const selectedIds = useMemo(() => selection.flatMap((item) => document && !isModelAddress(item) &&
    item.document === document ? [item.element.id] : []), [document, selection]);
  return {
    selectedId: selectedIds.length === 1 ? selectedIds[0] : null,
    selectedIds,
    /** Selects `id`, or with `extend` toggles it among the selected elements. */
    onSelect: (id: string | null, extend = false) => {
      if (!document) return;
      if (id === null) store.setState({ selection: [] });
      else selectElement(store, { document, element: { kind: 'named', id } }, extend);
    },
  };
}
