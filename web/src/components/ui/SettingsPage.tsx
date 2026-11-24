import { installSettings } from '@window-shell/browser/settings';
import { useId, useLayoutEffect, useRef, type ReactNode } from 'react';

/** A label and its control; the search field matches on the label and `search`. */
export function SettingRow({ label, search, children }: { label: string; search?: string; children: ReactNode }) {
  return (
    <div className="shell-settings__row" data-settings-search-text={search}>
      <span aria-hidden="true" className="shell-settings__row-label">{label}</span>
      <div className="shell-settings__row-control">{children}</div>
    </div>
  );
}

export interface SettingsSection {
  readonly id: string;
  readonly title: string;
  readonly rows: ReactNode;
}

/** A settings page on window-shell's layout: the section list with its search field, and the sections. */
export function SettingsPage({ label, searchLabel, sections }: {
  label: string; searchLabel: string; sections: readonly SettingsSection[];
}) {
  const root = useRef<HTMLDivElement>(null);
  const id = useId();
  useLayoutEffect(() => installSettings(root.current!).dispose, []);
  return (
    <div ref={root} className="shell-settings">
      <div className="shell-settings__layout">
        <nav className="shell-settings__nav" aria-label={label}>
          <input type="search" className="shell-settings__search" aria-label={searchLabel}
            placeholder={searchLabel} />
          {sections.map((section) => (
            <button key={section.id} type="button" className="shell-settings__link" aria-controls={`${id}-${section.id}`}>
              {section.title}
            </button>
          ))}
        </nav>
        <div className="shell-settings__content">
          {sections.map((section) => (
            <section key={section.id} id={`${id}-${section.id}`} className="shell-settings__section"
              aria-labelledby={`${id}-${section.id}-title`}>
              <h3 id={`${id}-${section.id}-title`} className="shell-settings__section-title">{section.title}</h3>
              {section.rows}
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
