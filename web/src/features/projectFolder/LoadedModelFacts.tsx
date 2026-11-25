import { studioName } from '@/components/activity/studioVocabulary';
import { FieldRow } from '@/components/ui/FieldRow';
import { Inspector } from '@/components/ui/Inspector';
import { shownName } from '@/features/pipeline/planView';
import { LoadedModelDocumentSchema } from '@/generated/studioContracts';
import type { AuthoredProject } from './authoredEdits';

const digest = (sha256: string) => <span className="font-mono">{sha256.slice(0, 12)}</span>;

/** What a Loaded Model document holds: the Model it fits, at the revision it was fitted to, and its weights. */
export function LoadedModelFacts({ project, document }: { project: AuthoredProject; document: string }) {
  const stored = project.documents?.[document];
  const parsed = stored && 'payload' in stored ? LoadedModelDocumentSchema.safeParse(stored.payload) : undefined;
  if (!parsed?.success) return null;
  const { model, source } = parsed.data;
  return <Inspector name={shownName(document, stored)} type={studioName('loaded_model')}>
    <FieldRow label={studioName('model')}>{shownName(model.document, project.documents?.[model.document])}</FieldRow>
    <FieldRow label={studioName('modelRevision')}>{digest(source.model.sha256)}</FieldRow>
    <FieldRow label={studioName('weights')}>{digest(source.weights.descriptor.sha256)}</FieldRow>
  </Inspector>;
}
