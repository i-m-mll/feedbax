import { useRef, useState } from 'react';
import { attachmentPoints } from '@/features/domains/planarAuthoring';
import { frameOptionsForGraph } from '@/features/domains/mechanicsAssembly';
import { planarSceneEntityId } from './planarModel';
import type { usePlanarSceneAuthoring } from './usePlanarSceneAuthoring';

type Authoring = ReturnType<typeof usePlanarSceneAuthoring>;

function NumericField({ label, value, apply }: { label: string; value: number; apply: (value: number) => void }) {
  const [text, setText] = useState<string | null>(null);
  const cancel = useRef(false);
  return (
    <label className="grid gap-1 text-sm">
      {label}
      <input type="number" step="any" value={text ?? value} className="rounded border px-2 py-1"
        onChange={(event) => setText(event.currentTarget.value)}
        onBlur={() => {
          if (!cancel.current && text !== null && text.trim() && Number.isFinite(Number(text))) apply(Number(text));
          cancel.current = false;
          setText(null);
        }}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            event.stopPropagation();
            cancel.current = true;
            event.currentTarget.blur();
          } else if (event.key === 'Enter') {
            event.preventDefault();
            event.currentTarget.blur();
          }
        }} />
    </label>
  );
}

export function ScenePlanarFields({ authoring, selectedId }: { authoring: Authoring; selectedId: string | null }) {
  const { editable, authored, source, view, apply, editIssue } = authoring;
  if (!editable || !authored || !source || view.kind !== 'geometry') return null;
  const selected = view.scene.anchors.find((item) => item.id === selectedId);
  const { muscle, point_index: index } = selected?.metadata ?? {};
  const entries = Object.entries(authored.nodes ?? {});
  const attachment = typeof muscle === 'string' && typeof index === 'number' ? entries.find(([id]) => id === muscle) : undefined;
  const point = attachment && typeof index === 'number' ? attachmentPoints(attachment[1])?.[index] : undefined;
  const entity = entries.find(([id]) => planarSceneEntityId(source, id) === selectedId);
  const [id, node] = entity ?? [];
  const link = source.model?.links.find((item) => item.body_id === id);
  const joint = source.model?.links.find((item) => item.joint_id === id);
  const length = Number(node?.params?.length ?? link?.length);
  const angle = Number(node?.params?.frame_angle ?? joint?.reference_angle);
  return (
    <div className="space-y-2">
      {link && id && <NumericField key={`${id}/length/${length}`} label="Length (m)" value={length}
        apply={(value) => { apply({ kind: 'set-link', node: id, field: 'length', value }); }} />}
      {joint && id && <NumericField key={`${id}/angle/${angle}`} label="Frame angle (rad)" value={angle}
        apply={(value) => { apply({ kind: 'set-joint', node: id, field: 'frame_angle', value }); }} />}
      {point && typeof index === 'number' && typeof muscle === 'string' && (
        <fieldset key={JSON.stringify([selectedId, point])} className="space-y-2">
          <label className="grid gap-1 text-sm">
            Attachment frame
            <select value={point.frame} className="rounded border px-2 py-1" onChange={(event) => {
              apply({ kind: 'set-attachment', node: muscle, index, expectedFrame: point.frame,
                attachment: { frame: event.currentTarget.value, offset: point.offset } });
            }}>
              {frameOptionsForGraph(authored).map((frame) => <option key={frame.id} value={frame.id}>{frame.id}</option>)}
            </select>
          </label>
          {(['X', 'Y'] as const).map((axis, coordinate) => (
            <NumericField key={axis} label={`${axis} offset (m)`} value={point.offset[coordinate]} apply={(value) => {
              const offset = [...point.offset] as [number, number];
              offset[coordinate] = value;
              apply({ kind: 'set-attachment', node: muscle, index, expectedFrame: point.frame, attachment: { frame: point.frame, offset } });
            }} />
          ))}
        </fieldset>
      )}
      {editIssue && <p role="alert" className="text-sm text-red-700">{editIssue}</p>}
    </div>
  );
}
