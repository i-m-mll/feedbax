import { Popover as BasePopover } from '@base-ui/react/popover';
import clsx from 'clsx';
import type { ReactNode } from 'react';
import { popupSurface } from './styles';
import { usePopupLayer } from './popupLayer';

export const Popover = BasePopover.Root;
export const PopoverTrigger = BasePopover.Trigger;
export const PopoverClose = BasePopover.Close;

export interface PopoverContentProps {
  children: ReactNode;
  side?: 'top' | 'bottom' | 'left' | 'right';
  align?: 'start' | 'center' | 'end';
  'aria-label'?: string;
  className?: string;
}

export function PopoverContent({ children, side = 'bottom', align = 'start', className, ...aria }: PopoverContentProps) {
  return (
    <BasePopover.Portal container={usePopupLayer()}>
      <BasePopover.Positioner className="z-50 outline-none" side={side} align={align} sideOffset={4}>
        <BasePopover.Popup className={clsx(popupSurface, 'p-3', className)} {...aria}>{children}</BasePopover.Popup>
      </BasePopover.Positioner>
    </BasePopover.Portal>
  );
}
