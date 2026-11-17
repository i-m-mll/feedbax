import clsx from 'clsx';
import { forwardRef, type HTMLAttributes } from 'react';

export interface CardProps extends HTMLAttributes<HTMLElement> {
  selected?: boolean;
  /** `article` when the card is a document element of its own, as a Model component is. */
  as?: 'div' | 'article';
}

/** The surface of a map step or a Model element; its look is the `--card-*` tokens. */
export const Card = forwardRef<HTMLElement, CardProps>(function Card({ selected, as: Element = 'div', className, ...props }, ref) {
  return <Element ref={ref as never} data-selected={selected || undefined} className={clsx('studio-card', className)} {...props} />;
});

export const CardHead = forwardRef<HTMLElement, HTMLAttributes<HTMLElement> & { as?: 'div' | 'header' }>(function CardHead(
  { as: Element = 'div', className, ...props }, ref,
) {
  return <Element ref={ref as never} className={clsx('studio-card__head', className)} {...props} />;
});
