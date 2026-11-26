import { describe, expect, it } from 'vitest';
import { DelayAttachmentSchema, InterventionAttachmentSchema, type ModelAddress } from '@/generated/studioContracts';
import { attachmentSources, type Attachment, type SourceValue } from './ModelInspector';

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
  const delay: Attachment = { kind: 'delay', id: 'd', delay: { value: 1, unit: 'st' } };
  const add: Attachment = { kind: 'intervention', id: 'i', operation: 'add', operands: { value: { kind: 'value', value: [0, 0] } } };

  // Catches: an Initial or an operand picked from a component output or an Input stored as another end, or one the Service refuses.
  it("stores a delay's Initial and an intervention's operand from a wire as the end it reads, and reads it back as the same port", () => {
    for (const item of [delay, add]) {
      for (const port of [output, input]) {
        const written = stored(item, { kind: 'wire', port });
        expect((item.kind === 'delay' ? DelayAttachmentSchema : InterventionAttachmentSchema).parse(written)).toEqual(written);
        const [read] = attachmentSources(written, graph).map((source) => source.read);
        expect(read).toEqual({ kind: 'wire', port });
      }
    }
  });

  // Catches: a Task variable read back as a value or a port, or a cleared operand left empty.
  it('stores a Task variable by its id, and Clear gives the operand its own zero again', () => {
    const varied = stored(add, { kind: 'task', variable: 'amplitude' });
    expect(attachmentSources(varied, graph)[0].read).toEqual({ kind: 'task', variable: 'amplitude' });
    const source = attachmentSources(varied, graph)[0];
    expect(stored(varied, { kind: 'value', value: source.unlinked })).toMatchObject({ operands: { value: { kind: 'value', value: [0] } } });
  });
});
