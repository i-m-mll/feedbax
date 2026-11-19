export const focusRing =
  'outline-none focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-1 focus-visible:outline-shell-accent';

export const glyph = '[&_svg]:size-[1.15em] [&_svg]:shrink-0';

export const disabledLook = 'data-[disabled]:cursor-default data-[disabled]:opacity-45';

export const fieldFrame =
  'flex rounded-control border border-shell-field bg-shell-surface text-shell-foreground ' +
  'focus-within:outline focus-within:outline-2 focus-within:-outline-offset-1 focus-within:outline-shell-accent ' +
  'has-[[aria-invalid=true]]:border-shell-status-failure has-[input:disabled]:opacity-45 has-[textarea:disabled]:opacity-45';

/** A field's leading glyph: a 1.5 em box 4 px inside the border, the same in every field type. */
export const leadingSlot = 'ml-1 flex size-[1.5em] shrink-0 items-center justify-center text-shell-muted ' + glyph;

export const fieldBox = 'h-control items-center ' + fieldFrame;

export const fieldInput =
  'h-full min-w-0 flex-1 bg-transparent px-control-inline text-[length:inherit] text-shell-foreground outline-none ' +
  'placeholder:text-shell-muted read-only:text-shell-muted disabled:cursor-default';

export const popupSurface =
  'rounded-surface border border-shell-divider bg-shell-chrome p-1 text-shell-foreground shadow-lift outline-none';

export const popupItem =
  'flex h-row cursor-default select-none items-center gap-2 rounded-row px-row-inline outline-none ' +
  'data-[highlighted]:bg-shell-hover data-[disabled]:opacity-45 ' + glyph;

export const primaryText = 'min-w-0 flex-auto truncate';

export const secondaryText = 'min-w-0 shrink-[1000] truncate pl-4 text-shell-muted';
