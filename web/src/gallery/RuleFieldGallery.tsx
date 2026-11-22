import { useState } from 'react';
import { RuleField } from '@/components/values/RuleField';
import type { RuleVocabulary } from '@/components/values/runRule';

const vocabulary: RuleVocabulary = {
  axes: { seed: ['0', '1', '2', '3'], lr: ['0.001', '0.01', '0.1', '1'] },
  metrics: ['loss', 'objective/total'],
};

/** A rule field over a 4 × 4 sweep: saved, refused, and with pinned rows. */
export function RuleFieldGallery() {
  const [rule, setRule] = useState('seed=0..1');
  return <div className="grid w-80 content-start gap-2 rounded-control border border-shell-divider p-2" aria-label="Rule fields">
    <RuleField label="Runs" value={rule} vocabulary={vocabulary} count="8/16" onCommit={setRule} />
    <RuleField label="Refused" value="sed=0" vocabulary={vocabulary} count="16/16" refusal="sed=0: sed is not swept in train"
      onCommit={() => undefined} />
    <RuleField label="Pinned" value="lowest 2 loss" vocabulary={vocabulary} count="2/16" pinned={['seed=0--lr=0.1', 'seed=2--lr=0.1']}
      onCommit={() => undefined} />
    <RuleField label="Exception" value="lr>=0.1" vocabulary={{ axes: vocabulary.axes }} count="8/16" onCommit={() => undefined} />
  </div>;
}
