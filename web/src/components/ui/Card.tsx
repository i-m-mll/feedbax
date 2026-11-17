import clsx from 'clsx';
import { forwardRef, type HTMLAttributes } from 'react';

/** What a card's members amount to, for the card styles that draw it on the card itself. */
export type CardStatus = 'busy' | 'failure' | 'success';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  selected?: boolean;
  status?: CardStatus;
}

/**
 * The surface of a canvas element: a map step, a Model element, an Analysis operation.
 * Its look is the `--card-*` tokens and the root's `data-card` style.
 */
export const Card = forwardRef<HTMLDivElement, CardProps>(function Card({ selected, status, className, ...props }, ref) {
  return <div ref={ref} data-selected={selected || undefined} data-status={status}
    className={clsx('studio-card', className)} {...props} />;
});

export const CardHead = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(function CardHead(
  { className, ...props }, ref,
) {
  return <div ref={ref} className={clsx('studio-card__head', className)} {...props} />;
});
