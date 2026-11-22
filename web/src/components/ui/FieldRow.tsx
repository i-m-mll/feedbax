import clsx from 'clsx';
import type { ReactNode } from 'react';
import './fieldRow.css';

export interface FieldRowProps {
  label: ReactNode;
  /** The id of the control the label names. Without it the label is text beside a group of controls. */
  htmlFor?: string;
  /** The label's own id, for controls that name themselves by `aria-labelledby`. */
  labelId?: string;
  children?: ReactNode;
  /** Commands on the field as a whole, at the label's far edge. */
  actions?: ReactNode;
  /** Lines under the field, in the field's column: its origin, its reason. */
  below?: ReactNode;
  className?: string;
}

/** An inspector row: a small muted label above its field. */
export function FieldRow({ label, htmlFor, labelId, children, actions, below, className }: FieldRowProps) {
  const name = htmlFor ? <label id={labelId} htmlFor={htmlFor} className="field-row-label">{label}</label>
    : <span id={labelId} className="field-row-label">{label}</span>;
  return (
    <div className={clsx('field-row', className)}>
      <div className="field-row-grid">
        {actions ? <div className="field-row-head">{name}<div className="field-row-actions">{actions}</div></div> : name}
        <div className="field-row-value">{children}</div>
        {below ? <div className="field-row-below">{below}</div> : null}
      </div>
    </div>
  );
}
