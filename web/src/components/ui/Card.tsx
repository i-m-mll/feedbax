import clsx from 'clsx';
import { forwardRef, type HTMLAttributes } from 'react';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  selected?: boolean;
}

/** The surface of a map step or a Model element; its look is the `--card-*` tokens. */
export const Card = forwardRef<HTMLDivElement, CardProps>(function Card({ selected, className, ...props }, ref) {
  return <div ref={ref} data-selected={selected || undefined} className={clsx('studio-card', className)} {...props} />;
});

export const CardHead = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(function CardHead(
  { className, ...props }, ref,
) {
  return <div ref={ref} className={clsx('studio-card__head', className)} {...props} />;
});
