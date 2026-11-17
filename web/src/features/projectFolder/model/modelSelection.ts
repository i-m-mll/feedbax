import { useState } from 'react';
import { createStore } from 'zustand/vanilla';
import type { ModelAddress } from '@/generated/studioContracts';
import { addressKey, type Port } from './modelGraph';

type Pick = { accepts(address: ModelAddress, port: Port): boolean; choose(address: ModelAddress): void };
type State = { selection: readonly ModelAddress[]; pick: Pick | null; reveal: ModelAddress | null };

export type ModelSelectionStore = ReturnType<typeof useModelSelectionStore>;
export function useModelSelectionStore() {
  const [store] = useState(() => createStore<State>(() => ({ selection: [], pick: null, reveal: null })));
  return store;
}

export function selectModel(store: ModelSelectionStore, address: ModelAddress, extend = false) {
  const current = store.getState().selection;
  if (!extend || current.some((item) => item.document !== address.document ||
      JSON.stringify(item.composites) !== JSON.stringify(address.composites) || item.element.kind !== address.element.kind)) {
    store.setState({ selection: [address] });
  } else {
    const key = addressKey(address);
    const other = current.filter((item) => addressKey(item) !== key);
    store.setState({ selection: other.length === current.length ? [...other, address] : other });
  }
}
