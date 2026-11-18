import clsx from 'clsx';
import { forwardRef, type CSSProperties, type HTMLAttributes } from 'react';

/** What a card's members amount to, for the card styles that draw it on the card itself. */
export type CardStatus = 'busy' | 'failure' | 'success';

export interface CardProps extends HTMLAttributes<HTMLElement> {
  selected?: boolean;
  status?: CardStatus;
  /** The border colour of a card that has a kind; a card without one has the neutral border. */
  tint?: string;
  /** `article` when the card is a document element of its own, as a Model component is. */
  as?: 'div' | 'article';
}

/**
 * The surface of a canvas element: a map step, a Model element, an Analysis operation.
 * Its look is the `--card-*` tokens and the root's `data-card` style.
 */
export const Card = forwardRef<HTMLElement, CardProps>(function Card(
  { selected, status, tint, as: Element = 'div', className, style, ...props }, ref,
) {
  return <Element ref={ref as never} data-selected={selected || undefined} data-status={status}
    className={clsx('studio-card', className)} style={tint ? { ...cardTint(tint), ...style } : style} {...props} />;
});

/** The style that gives an element drawn as a card its kind's border, for one that is not a `Card`. */
export function cardTint(tint: string): CSSProperties {
  return { '--card-tint': tint } as CSSProperties;
}

export const CardHead = forwardRef<HTMLElement, HTMLAttributes<HTMLElement> & { as?: 'div' | 'header' }>(function CardHead(
  { as: Element = 'div', className, ...props }, ref,
) {
  return <Element ref={ref as never} className={clsx('studio-card__head', className)} {...props} />;
});
