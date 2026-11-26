import { describe, expect, it } from 'vitest';
import { StoredDelayAttachmentSchema, StoredInterventionAttachmentSchema, type ModelAddress } from '@/generated/studioContracts';
import { attachmentFields, attachmentSources, type SourceValue } from './ModelInspector';
import { storedAttachment, type Attachment } from './modelGraph';

const graph: ModelAddress = { composites: ['core'], element: { kind: 'graph' } };
const output: ModelAddress = { composites: ['core'], element: { kind: 'port', end: { owner: 'component', component: 'enc', port: 'out', direction: 'output' } } };
const input: ModelAddress = { composites: ['core'], element: { kind: 'port', end: { owner: 'inputs', port: 'goal' } } };

/** What one source of `item` stores when set to `value`. */
function stored(item: Attachment, value: SourceValue, index = 0) {
  let written: Attachment | undefined;
  void attachmentSources(item, graph, (next) => { written = next; })[index].set!(value);
  return written!;
}

describe('attachmentSources', () => {
  const delay: Attachment = { kind: 'delay', id: 'd', delay: '1 st' };
  const add: Attachment = { kind: 'intervention', id: 'i', operation: 'add', operands: { value: { kind: 'value', value: '[0, 0]' } } };

  // Catches: an Initial or an operand picked from a component output or an Input stored as another end, or one the Service refuses.
  it("stores a delay's Initial and an intervention's operand from a wire as the end it reads, and reads it back as the same port", () => {
    for (const item of [delay, add]) {
      for (const port of [output, input]) {
        const written = stored(item, { kind: 'wire', port });
        const kept = storedAttachment(written);
        expect((item.kind === 'delay' ? StoredDelayAttachmentSchema : StoredInterventionAttachmentSchema).parse(kept)).toEqual(kept);
        const [read] = attachmentSources(written, graph).map((source) => source.read);
        expect(read).toEqual({ kind: 'wire', port });
      }
    }
  });

  // Catches: a Task variable read back as a value or a port.
  it('stores a Task variable by its id', () => {
    const varied = stored(add, { kind: 'task', variable: 'amplitude' });
    expect(attachmentSources(varied, graph)[0].read).toEqual({ kind: 'task', variable: 'amplitude' });
  });

  // Catches: a cleared Initial stored as a literal of Studio's own instead of the Service's default.
  it("clears a delay's Initial by storing none", () => {
    let written: Attachment | undefined;
    const linked: Attachment = { ...delay, initial_source: { kind: 'task', variable: 'amplitude' } };
    void attachmentSources(linked, graph, (next) => { written = next; })[0].clear!();
    expect(written).toEqual(delay);
  });

  // Catches: another operation stored with the operands of the one it replaced, which the Service would keep.
  it('stores another operation without operands, so the Service gives its own', async () => {
    let written: Attachment | undefined;
    const [operation] = attachmentFields(add, (next) => { written = next; });
    await operation.commit!({ kind: 'set', value: 'clamp' });
    expect(written).toEqual({ kind: 'intervention', id: 'i', operation: 'clamp' });
  });
});
