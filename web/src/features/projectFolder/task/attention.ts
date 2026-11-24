/**
 * A control that acts on one row, column or cell (P32): drawn while the pointer is over its owner, focus is inside it,
 * or it is selected, and keeping its place otherwise. The owner carries `group/owner`.
 */
export const withAttention = 'opacity-0 group-hover/owner:opacity-100 group-focus-within/owner:opacity-100 group-data-[selected]/owner:opacity-100 focus-visible:opacity-100';
