import clsx from 'clsx';
import { Plus } from 'lucide-react';
import type { ReactNode } from 'react';
import { ListRow } from './ListRow';
import { Menu, MenuContent, MenuTrigger } from './Menu';

export interface InspectorProps {
  /** The element's name. */
  name: ReactNode;
  /** The element's type, left out when it would repeat the name. */
  type?: ReactNode;
  /** The element's fields, as rows. */
  children?: ReactNode;
  /** The element's attachments, as rows of the tree's format, after a hairline. */
  attachments?: ReactNode;
  className?: string;
}

/** The one inspector layout: a title line, the fields as rows, a hairline, the attachments. */
export function Inspector({ name, type, children, attachments, className }: InspectorProps) {
  return (
    <section className={clsx('grid min-w-0 content-start gap-2 p-2 text-sm', className)}>
      <header className="flex h-row min-w-0 items-baseline gap-2">
        <span className="min-w-0 truncate font-medium">{name}</span>
        {type ? <span className="min-w-0 truncate text-shell-muted">{type}</span> : null}
      </header>
      {children ? <div className="grid min-w-0 gap-1">{children}</div> : null}
      {attachments ? <div className="grid min-w-0 border-t border-shell-divider pt-1">{attachments}</div> : null}
    </section>
  );
}

/** The `+` row that ends an inspector's attachments; its items are the attachments that can be added. */
export function InspectorAddRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Menu>
      <MenuTrigger nativeButton={false} render={<ListRow role="button" tabIndex={0} aria-label={label} glyph={<Plus />} name="" />} />
      <MenuContent aria-label={label}>{children}</MenuContent>
    </Menu>
  );
}
