import { useState } from 'react';
import { createStore } from 'zustand/vanilla';
import type { ModelAddress } from '@/generated/studioContracts';
import { addressKey, type Port } from './modelGraph';

export type ModelLocation = { document: string; address: ModelAddress };
export type ModelSelection = { document: string; addresses: readonly ModelAddress[] };
type Pick = { document: string; accepts(address: ModelAddress, port: Port): boolean; choose(address: ModelAddress): void };
type State = { selection: ModelSelection | null; pick: Pick | null; reveal: ModelLocation | null };

export type ModelSelectionStore = ReturnType<typeof useModelSelectionStore>;
export function useModelSelectionStore() {
  const [store] = useState(() => createStore<State>(() => ({ selection: null, pick: null, reveal: null })));
  return store;
}

export function selectModel(store: ModelSelectionStore, document: string, address: ModelAddress, extend = false) {
  const current = store.getState().selection;
  if (!extend || current?.document !== document || current.addresses.some((item) =>
      JSON.stringify(item.composites) !== JSON.stringify(address.composites) || item.element.kind !== address.element.kind)) {
    store.setState({ selection: { document, addresses: [address] } });
  } else {
    const key = addressKey(address);
    const other = current.addresses.filter((item) => addressKey(item) !== key);
    store.setState({ selection: { document, addresses: other.length === current.addresses.length ? [...other, address] : other } });
  }
}
