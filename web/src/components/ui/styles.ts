export const focusRing =
  'outline-none focus-visible:outline focus-visible:outline-2 focus-visible:-outline-offset-1 focus-visible:outline-shell-accent';

export const glyph = '[&_svg]:size-[1.15em] [&_svg]:shrink-0';

export const disabledLook = 'data-[disabled]:cursor-default data-[disabled]:opacity-45';

export const fieldFrame =
  'flex rounded-control border border-shell-control bg-shell-surface text-shell-foreground ' +
  'focus-within:outline focus-within:outline-2 focus-within:-outline-offset-1 focus-within:outline-shell-accent ' +
  'has-[[aria-invalid=true]]:border-shell-status-failure has-[input:disabled]:opacity-45 has-[textarea:disabled]:opacity-45';

export const fieldBox = 'h-control items-center ' + fieldFrame;

export const fieldInput =
  'h-full min-w-0 flex-1 bg-transparent px-control-inline text-[length:inherit] text-shell-foreground outline-none ' +
  'placeholder:text-shell-muted read-only:text-shell-muted disabled:cursor-default';

export const popupSurface =
  'rounded-surface border border-shell-divider bg-shell-chrome p-1 text-shell-foreground shadow-lift outline-none';

export const popupItem =
  'flex h-row cursor-default select-none items-center gap-2 rounded-row px-row-inline outline-none ' +
  'data-[highlighted]:bg-shell-hover data-[disabled]:opacity-45 ' + glyph;
