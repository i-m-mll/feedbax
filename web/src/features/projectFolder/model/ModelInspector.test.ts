import { describe, expect, it } from 'vitest';
import { DelayAttachmentSchema, InterventionAttachmentSchema, type JsonValue, type ModelAddress } from '@/generated/studioContracts';
import { addressKey } from './modelGraph';
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
  const add: Attachment = { kind: 'intervention', id: 'i', operation: 'add', fields: { value: [0, 0] } };

  // Catches: an Initial picked from a component output or an Input stored as another end, or one the Service refuses.
  it("stores a delay's Initial from a wire as the end it reads, and reads it back as the same port", () => {
    for (const port of [output, input]) {
      const written = stored(delay, { kind: 'wire', port });
      expect(DelayAttachmentSchema.parse(written)).toEqual(written);
      const [read] = attachmentSources(written, graph).map((source) => source.read);
      expect(read).toEqual({ kind: 'wire', port });
    }
    expect(stored(delay, { kind: 'task', variable: 'h0' })).toMatchObject({ initial_source: { kind: 'task', variable: 'h0' } });
  });

  // Catches: a field reading a port read back as a Task variable (or the reverse), or a cleared field left empty.
  it("stores an intervention field's port by its address and a Task variable by its id, each read back as itself", () => {
    const wired = stored(add, { kind: 'wire', port: output });
    expect(InterventionAttachmentSchema.parse(wired)).toEqual(wired);
    expect((wired as { fields: Record<string, JsonValue> }).fields.value).toEqual({ kind: 'name', reference: addressKey(output) });
    expect(attachmentSources(wired, graph)[0].read).toEqual({ kind: 'wire', port: output });
    const varied = stored(add, { kind: 'task', variable: 'amplitude' });
    expect(attachmentSources(varied, graph)[0].read).toEqual({ kind: 'task', variable: 'amplitude' });
    const source = attachmentSources(varied, graph)[0];
    expect(stored(varied, { kind: 'value', value: source.unlinked })).toMatchObject({ fields: { value: [0] } });
  });
});
