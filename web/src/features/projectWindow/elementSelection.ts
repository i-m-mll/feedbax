import { useEffect, useMemo, useRef, useState } from 'react';
import { useStore } from 'zustand';
import { createStore } from 'zustand/vanilla';
import type { ModelAddress } from '@/generated/studioContracts';
import { addressKey, at, type Port } from '@/features/projectFolder/model/modelGraph';

/** An element of a document that has no element grammar of its own yet: an Analysis operation or figure, an Objective term. */
export interface NamedElementAddress {
  readonly document: string;
  readonly element: { readonly kind: 'named'; readonly id: string };
}

/** What the window's selection holds: the address of an element of a document. */
export type ElementAddress = ModelAddress | NamedElementAddress;

type Pick = { accepts(address: ModelAddress, port: Port): boolean; choose(address: ModelAddress): void };
type State = {
  selection: readonly ElementAddress[];
  /** A reference field's crosshair: the next element clicked fills it instead of being selected. */
  pick: Pick | null;
  /** A Model address the slot showing its document moves to. */
  reveal: ModelAddress | null;
};

export type ElementSelectionStore = ReturnType<typeof useElementSelectionStore>;

/** The window's one selection of elements, read by every tree, canvas, table and inspector. */
export function useElementSelectionStore() {
  const [store] = useState(() => createStore<State>(() => ({ selection: [], pick: null, reveal: null })));
  return store;
}

export function isModelAddress(address: ElementAddress): address is ModelAddress {
  return address.element.kind !== 'named';
}

/** The element a selected address stands for: a field is shown as the element that holds it. */
export function selectedElement(address: ModelAddress): ModelAddress {
  return address.element.kind === 'field' ? at(address, address.element.parent) : address;
}

export function elementKey(address: ElementAddress) {
  return isModelAddress(address) ? addressKey(address) : JSON.stringify([address.document, address.element.id]);
}

function sameGroup(left: ElementAddress, right: ElementAddress) {
  if (left.document !== right.document || left.element.kind !== right.element.kind) return false;
  return !isModelAddress(left) || !isModelAddress(right) ||
    JSON.stringify(left.composites) === JSON.stringify(right.composites);
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

export function clearSelection(store: ElementSelectionStore) {
  store.setState({ selection: [] });
}

/** The selected Model elements, fields resolved to the element that holds them. */
export function useModelSelection(store: ElementSelectionStore) {
  const selection = useStore(store, (state) => state.selection);
  return useMemo(() => selection.filter(isModelAddress).map(selectedElement), [selection]);
}

/**
 * The selected element of one document without an element grammar, as one id. `reveal`, when it changes, selects
 * `revealedId`.
 */
export function useNamedSelection(store: ElementSelectionStore, document: string | undefined, reveal?: object, revealedId?: string) {
  const selection = useStore(store, (state) => state.selection);
  const handled = useRef<object>();
  useEffect(() => {
    if (!document || !reveal || revealedId === undefined || handled.current === reveal) return;
    handled.current = reveal;
    store.setState({ selection: [{ document, element: { kind: 'named', id: revealedId } }] });
  }, [document, reveal, revealedId, store]);
  const first = selection[0];
  return {
    selectedId: document && first && !isModelAddress(first) && first.document === document ? first.element.id : null,
    onSelect: (id: string | null) => {
      if (!document) return;
      store.setState({ selection: id === null ? [] : [{ document, element: { kind: 'named', id } }] });
    },
  };
}
