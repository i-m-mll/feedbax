import type { LucideIcon } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { appTip } from '@window-shell/browser/tooltip';
import { StudioRegionBoundary } from '@/components/activity/StudioRegion';
import { studioName } from '@/components/activity/studioVocabulary';
import { DocumentPicker, type DocumentEditing, type DocumentPickerModel } from './DocumentPicker';
import type { ProjectWindowView } from './projectWindowState';

/** A slot's header: the view's glyph in split view, the picker of the document it shows, the path or controls inside it. */
export function SlotHeader({ view, icon: Icon, picker, children, trailing }: {
  view: ProjectWindowView;
  /** Drawn only when another slot shares the window; alone, the rail already shows the view. */
  icon?: LucideIcon;
  picker?: DocumentPickerModel;
  children?: ReactNode;
  trailing: ReactNode;
}) {
  const name = studioName(view);
  const [editing, setEditing] = useState<DocumentEditing>();
  return <header className="studio-window-slot__header">
    {Icon
      ? <h2 className="studio-window-slot__name" {...appTip(studioName(view))}>
          <Icon aria-hidden="true" />
          <span className="sr-only">{name}</span>
        </h2>
      : <h2 className="sr-only">{name}</h2>}
    <div className="studio-window-slot__controls">
      <StudioRegionBoundary key={view} name={view}>
        {picker && <DocumentPicker model={picker} editing={editing} setEditing={setEditing} />}
        {children}
      </StudioRegionBoundary>
    </div>
    {trailing}
  </header>;
}
