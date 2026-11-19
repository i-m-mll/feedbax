import { ContextMenu as BaseContextMenu } from '@base-ui/react/context-menu';
import { Menu as BaseMenu } from '@base-ui/react/menu';
import clsx from 'clsx';
import { Check } from 'lucide-react';
import { useRef, type ComponentPropsWithoutRef, type ReactNode } from 'react';
import { popupItem, popupSurface, primaryText, secondaryText } from './styles';
import { usePopupLayer } from './popupLayer';

export const Menu = BaseMenu.Root;
export const MenuTrigger = BaseMenu.Trigger;
export const ContextMenu = BaseContextMenu.Root;
export const ContextMenuTrigger = BaseContextMenu.Trigger;

type Side = 'top' | 'bottom' | 'left' | 'right' | 'inline-start' | 'inline-end';

type PositionerProps = ComponentPropsWithoutRef<typeof BaseMenu.Positioner>;
type PopupProps = ComponentPropsWithoutRef<typeof BaseMenu.Popup>;

export interface MenuContentProps {
  children: ReactNode;
  side?: Side;
  align?: 'start' | 'center' | 'end';
  anchor?: PositionerProps['anchor'];
  finalFocus?: PopupProps['finalFocus'];
  'aria-label'?: string;
  className?: string;
}

export function MenuContent({ children, side = 'bottom', align = 'start', anchor, finalFocus, className, ...aria }: MenuContentProps) {
  const popup = useRef<HTMLDivElement>(null);
  // Focus returns when the close animation ends. If the person has moved focus elsewhere by then, it stays there:
  // returning it would send their next keys to the field the menu came from.
  const settle: PopupProps['finalFocus'] = (closeType) => {
    const active = document.activeElement;
    if (active && active !== document.body && !popup.current?.contains(active)) return false;
    if (typeof finalFocus === 'function') return finalFocus(closeType);
    return typeof finalFocus === 'object' ? finalFocus.current : finalFocus ?? true;
  };
  return (
    <BaseMenu.Portal container={usePopupLayer()}>
      <BaseMenu.Positioner className="z-50 outline-none" side={side} align={align} sideOffset={4} anchor={anchor}>
        <BaseMenu.Popup ref={popup} className={clsx(popupSurface, 'min-w-[10rem]', className)} finalFocus={settle} {...aria}>{children}</BaseMenu.Popup>
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
      <span className={primaryText}>{children}</span>
      {shortcut ? <span className={secondaryText}>{shortcut}</span> : null}
    </BaseMenu.Item>
  );
}

export interface MenuCheckboxItemProps {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
  children: ReactNode;
}

export function MenuCheckboxItem({ checked, onCheckedChange, disabled, children }: MenuCheckboxItemProps) {
  return (
    <BaseMenu.CheckboxItem className={popupItem} checked={checked} disabled={disabled} closeOnClick={false}
      onCheckedChange={(next) => onCheckedChange(next)}>
      <span className="flex size-[1em] shrink-0 items-center justify-center">
        <BaseMenu.CheckboxItemIndicator keepMounted={false}><Check /></BaseMenu.CheckboxItemIndicator>
      </span>
      <span className={primaryText}>{children}</span>
    </BaseMenu.CheckboxItem>
  );
}

export function MenuSeparator() {
  return <BaseMenu.Separator className="mx-1 my-1 h-px bg-shell-divider" />;
}

export function MenuGroupLabel({ children }: { children: ReactNode }) {
  return <BaseMenu.GroupLabel className="flex h-row items-center px-row-inline text-sm text-shell-muted">{children}</BaseMenu.GroupLabel>;
}

export const MenuGroup = BaseMenu.Group;

export const MenuRadioGroup = BaseMenu.RadioGroup;

export function MenuRadioItem({ value, disabled, children }: { value: string; disabled?: boolean; children: ReactNode }) {
  return (
    <BaseMenu.RadioItem className={popupItem} value={value} disabled={disabled} closeOnClick>
      <span className="flex size-[1em] shrink-0 items-center justify-center">
        <BaseMenu.RadioItemIndicator keepMounted={false}><Check /></BaseMenu.RadioItemIndicator>
      </span>
      <span className={primaryText}>{children}</span>
    </BaseMenu.RadioItem>
  );
}
