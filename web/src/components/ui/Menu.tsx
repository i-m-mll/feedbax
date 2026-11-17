import { ContextMenu as BaseContextMenu } from '@base-ui/react/context-menu';
import { Menu as BaseMenu } from '@base-ui/react/menu';
import clsx from 'clsx';
import type { ComponentProps, ComponentPropsWithoutRef, ReactNode } from 'react';
import { popupItem, popupSurface } from './styles';
import { usePopupLayer } from './popupLayer';

export const Menu = BaseMenu.Root;
export const MenuTrigger = BaseMenu.Trigger;
export const ContextMenu = BaseContextMenu.Root;
export const ContextMenuTrigger = BaseContextMenu.Trigger;

type Side = 'top' | 'bottom' | 'left' | 'right' | 'inline-start' | 'inline-end';

export interface MenuContentProps {
  children: ReactNode;
  side?: Side;
  align?: 'start' | 'center' | 'end';
  /** Positions the menu against this instead of its trigger. */
  anchor?: ComponentProps<typeof BaseMenu.Positioner>['anchor'];
  finalFocus?: ComponentProps<typeof BaseMenu.Popup>['finalFocus'];
  className?: string;
}

export function MenuContent({ children, side = 'bottom', align = 'start', anchor, finalFocus, className }: MenuContentProps) {
  return (
    <BaseMenu.Portal container={usePopupLayer()}>
      <BaseMenu.Positioner className="z-50 outline-none" side={side} align={align} sideOffset={4} anchor={anchor}>
        <BaseMenu.Popup className={clsx(popupSurface, 'min-w-[10rem]', className)} finalFocus={finalFocus}>{children}</BaseMenu.Popup>
      </BaseMenu.Positioner>
    </BaseMenu.Portal>
  );
}

export function ContextMenuContent({ children, className }: Pick<MenuContentProps, 'children' | 'className'>) {
  return (
    <BaseContextMenu.Portal container={usePopupLayer()}>
      <BaseContextMenu.Positioner className="z-50 outline-none">
        <BaseContextMenu.Popup className={clsx(popupSurface, 'min-w-[10rem]', className)}>{children}</BaseContextMenu.Popup>
      </BaseContextMenu.Positioner>
    </BaseContextMenu.Portal>
  );
}

export interface MenuItemProps extends Omit<ComponentPropsWithoutRef<typeof BaseMenu.Item>, 'className' | 'children'> {
  glyph?: ReactNode;
  shortcut?: string;
  children: ReactNode;
  className?: string;
}

export function MenuItem({ glyph, shortcut, children, className, ...props }: MenuItemProps) {
  return (
    <BaseMenu.Item className={clsx(popupItem, className)} {...props}>
      {glyph}
      <span className="flex-1 truncate">{children}</span>
      {shortcut ? <span className="pl-4 text-shell-muted">{shortcut}</span> : null}
    </BaseMenu.Item>
  );
}

export function MenuSeparator() {
  return <BaseMenu.Separator className="mx-1 my-1 h-px bg-shell-divider" />;
}

export function MenuGroupLabel({ children }: { children: ReactNode }) {
  return <BaseMenu.GroupLabel className="flex h-row items-center px-row-inline text-sm text-shell-muted">{children}</BaseMenu.GroupLabel>;
}

export const MenuGroup = BaseMenu.Group;
