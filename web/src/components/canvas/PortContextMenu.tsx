import { useTrainingStore } from '@/stores/trainingStore';
import type { LossTermSpec } from '@/types/training';
import { useCallback, useEffect, useRef } from 'react';
import { Crosshair, Plus } from 'lucide-react';

interface PortContextMenuProps {
  x: number;
  y: number;
  nodeName: string;
  portName: string;
  portType: 'input' | 'output';
  onClose: () => void;
}

export function PortContextMenu({
  x,
  y,
  nodeName,
  portName,
  portType,
  onClose,
}: PortContextMenuProps) {
  const addLossTerm = useTrainingStore((state) => state.addLossTerm);
  const trainingSpec = useTrainingStore((state) => state.trainingSpec);
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [onClose]);

  // Close on escape
  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [onClose]);

  const handleAddProbe = useCallback(() => {
    // Generate a unique key for the new loss term
    const baseKey = `${nodeName}_${portName}`.toLowerCase().replace(/[^a-z0-9]/g, '_');
    let key = baseKey;
    let counter = 1;

    // Find existing keys to avoid collision
    const existingKeys = new Set<string>();
    const collectKeys = (term: LossTermSpec) => {
      if (term.children) {
        Object.keys(term.children).forEach((k) => {
          existingKeys.add(k);
          collectKeys(term.children![k]);
        });
      }
    };
    collectKeys(trainingSpec.loss);

    while (existingKeys.has(key)) {
      key = `${baseKey}_${counter}`;
      counter++;
    }

    // Create the new loss term
    const newTerm: LossTermSpec = {
      type: 'TargetStateLoss',
      label: `${nodeName} ${portName}`,
      weight: 1.0,
      selector: `port:${nodeName}.${portName}`,
      norm: 'squared_l2',
      time_agg: {
        mode: 'all',
      },
    };

    // Add to the root loss term (as a child of the composite)
    addLossTerm([], key, newTerm);
    onClose();
  }, [nodeName, portName, trainingSpec.loss, addLossTerm, onClose]);

  // Only show for output ports
  if (portType !== 'output') {
    return null;
  }

  return (
    <div
      ref={menuRef}
      className="fixed bg-white rounded-lg shadow-lg border border-slate-200 py-1 min-w-40 z-50"
      style={{ left: x, top: y }}
    >
      <button
        type="button"
        onClick={handleAddProbe}
        className="w-full px-3 py-2 text-left text-sm text-slate-700 hover:bg-slate-50 flex items-center gap-2"
      >
        <Crosshair className="w-4 h-4 text-brand-500" />
        Add probe here
      </button>
      <div className="border-t border-slate-100 my-1" />
      <button
        type="button"
        onClick={onClose}
        className="w-full px-3 py-2 text-left text-sm text-slate-500 hover:bg-slate-50"
      >
        Cancel
      </button>
    </div>
  );
}
