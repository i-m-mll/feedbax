import { shapeFunctions, type CallDraft, type FieldModel } from './fieldModel';
import { CallParts } from './FormulaParts';

/** A Shape's parts: its function among the site's shapes, and its parameters. */
export function ShapeParts({ model, call }: { model: FieldModel; call: CallDraft }) {
  return <CallParts model={model} call={call} offered={shapeFunctions(model)} kind="shape" />;
}
