import clsx from 'clsx';
import { ChevronRight } from 'lucide-react';
import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { ScrollFade } from './ScrollFade';
import { focusRing, glyph } from './styles';

/**
 * A folding header's place in the hierarchy. A `drawer` section is one fold of a `SectionStack` and scrolls its own body;
 * a `group` (a group inside a list) and a `field` (a field group in a form) are plain blocks in their parent's body.
 */
export type SectionLevel = 'drawer' | 'group' | 'field';

export interface SectionProps {
  name: ReactNode;
  /** Id for the name, for a body that is labelled by it. */
  nameId?: string;
  level?: SectionLevel;
  /** Trailing commands, drawn at the header's far edge. */
  actions?: ReactNode;
  /** Drawn under the header and above the body, outside the body's scrolling, as a search field. */
  lead?: ReactNode;
  /** Drawn under the header whether or not the section is open: a field group's own control, as a class's picker. */
  control?: ReactNode;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

const header: Record<SectionLevel, string> = {
  drawer: 'border-b border-shell-divider bg-shell-header text-sm font-medium text-shell-foreground',
  group: 'text-sm text-shell-category',
  field: 'text-xs text-shell-muted',
};

export function Section({
  name, nameId, level = 'drawer', actions, lead, control, open: controlled, defaultOpen = true, onOpenChange, children, className, bodyClassName,
}: SectionProps) {
  const [uncontrolled, setUncontrolled] = useState(defaultOpen);
  const open = controlled ?? uncontrolled;
  const body = useId();
  const generated = useId();
  const title = nameId ?? generated;
  const toggle = () => {
    if (controlled === undefined) setUncontrolled(!open);
    onOpenChange?.(!open);
  };
  const content = <div className={bodyClassName}>{children}</div>;
  return (
    <section
      aria-labelledby={title}
      data-section={level}
      data-open={open || undefined}
      className={clsx(
        'flex min-w-0 shrink-0 flex-col',
        level === 'drawer' && '[[data-section=drawer][data-open]+&]:border-t [[data-section=drawer][data-open]+&]:border-shell-divider',
        className,
      )}
    >
      <div data-section-header className={clsx('flex h-row shrink-0 items-center gap-0.5', level !== 'field' && 'px-1', header[level], glyph)}>
        <button
          type="button"
          aria-expanded={open}
          aria-controls={body}
          onClick={toggle}
          className={clsx(
            'flex h-full min-w-0 flex-1 cursor-default select-none items-center gap-1 px-1 text-left hover:text-shell-foreground',
            level === 'field' && '-ml-1',
            focusRing,
          )}
        >
          <span id={title} className="min-w-0 truncate">{name}</span>
          <ChevronRight aria-hidden data-section-disclosure
            className={clsx('text-shell-muted transition-transform', open && 'rotate-90')} />
        </button>
        {actions ? <div className="flex shrink-0 items-center gap-0.5">{actions}</div> : null}
      </div>
      {control}
      {lead && open ? <div className="shrink-0">{lead}</div> : null}
      {level === 'drawer'
        ? <ScrollFade id={body} hidden={!open} data-section-body className="flex-none">{content}</ScrollFade>
        : <div id={body} hidden={!open}>{content}</div>}
    </section>
  );
}

/**
 * An open section's body: its content's height and, for a list, the heights at which the body ends on a row's bottom
 * edge, ascending and ending at the content's own.
 */
export interface FoldBody {
  content: number;
  stops?: readonly number[];
}

/** Each takes an equal share of `available`; one whose content is shorter takes only its content, giving the rest to the others. */
function fill(contents: readonly number[], available: number) {
  const heights = contents.map(() => 0);
  let remaining = Math.max(0, available);
  let active = contents.map((_, index) => index);
  while (active.length) {
    const share = remaining / active.length;
    const fitting = active.filter((index) => contents[index] <= share);
    if (!fitting.length) {
      for (const index of active) heights[index] = share;
      return { heights, remaining: 0 };
    }
    for (const index of fitting) {
      heights[index] = contents[index];
      remaining -= contents[index];
    }
    active = active.filter((index) => contents[index] > share);
  }
  return { heights, remaining };
}

/**
 * Heights for the open sections of a stack, given their bodies and the height they share: each takes an equal share,
 * one whose content is shorter takes only its content and gives the rest to those that overflow. A list that overflows
 * shows whole rows: its height ends on a row, and what that frees goes row by row first to the lists it lets show all
 * their rows, then to the lists that show least, then to bodies that are not lists. What nothing needs goes to the
 * last, unless the last is a list it would cut into; it is then the `rest`, a gap below the last.
 */
export function shareHeight(bodies: readonly FoldBody[], available: number): { heights: number[]; rest: number } {
  const contents = bodies.map(({ content }) => content);
  const { heights } = fill(contents, available);
  const listed = (index: number) => bodies[index].stops !== undefined;
  const short = (index: number) => heights[index] < contents[index];
  bodies.forEach(({ stops }, index) => {
    if (stops && short(index)) heights[index] = stops.filter((stop) => stop <= heights[index]).at(-1) ?? 0;
  });
  let free = Math.max(0, available) - heights.reduce((sum, height) => sum + height, 0);
  for (;;) {
    const next = (index: number) => bodies[index].stops!.find((stop) => stop > heights[index])!;
    const growing = bodies.map((_, index) => index)
      .filter((index) => listed(index) && short(index) && next(index) - heights[index] <= free)
      .sort((a, b) => Number(next(b) === contents[b]) - Number(next(a) === contents[a]) || heights[a] - heights[b]);
    if (!growing.length) break;
    const [index] = growing;
    free -= next(index) - heights[index];
    heights[index] = next(index);
  }
  const plain = bodies.map((_, index) => index).filter((index) => !listed(index) && short(index));
  const grown = fill(plain.map((index) => contents[index] - heights[index]), free);
  plain.forEach((index, at) => { heights[index] += grown.heights[at]; });
  free = plain.length ? grown.remaining : free;
  const last = heights.length - 1;
  if (last < 0) return { heights, rest: 0 };
  if (listed(last) && short(last)) return { heights, rest: free };
  heights[last] += free;
  return { heights, rest: 0 };
}

/** Where a list's body can end: on the bottom edge of each row but the last, and at the end of its content. */
function stopsOf(content: HTMLElement): readonly number[] | undefined {
  const top = content.getBoundingClientRect().top;
  const bottoms = [...content.querySelectorAll('[data-list-row]')]
    .map((row) => row.getBoundingClientRect())
    .filter(({ height }) => height > 0)
    .map(({ bottom }) => bottom - top);
  if (!bottoms.length) return undefined;
  // The last row ends the list with the content's own end, so its padding is never all that scrolls.
  const end = content.getBoundingClientRect().height;
  return [...new Set([...bottoms.slice(0, -1).filter((bottom) => bottom < end), end])].sort((a, b) => a - b);
}

function layout(stack: HTMLElement) {
  const open: { section: HTMLElement; body: HTMLElement; fold: FoldBody }[] = [];
  let fixed = 0;
  for (const section of stack.children) {
    if (!(section instanceof HTMLElement) || section.dataset.section !== 'drawer') continue;
    const body = section.querySelector<HTMLElement>(':scope > [data-section-body]');
    if (!body) continue;
    section.style.removeProperty('margin-bottom');
    fixed += section.offsetHeight - body.offsetHeight;
    const content = body.firstElementChild as HTMLElement | null;
    if (section.dataset.open === undefined) body.style.removeProperty('height');
    else open.push({ section, body, fold: content
      ? { content: content.getBoundingClientRect().height, stops: stopsOf(content) } : { content: 0 } });
  }
  const { heights, rest } = shareHeight(open.map(({ fold }) => fold), stack.clientHeight - fixed);
  open.forEach(({ body }, index) => { body.style.height = `${heights[index]}px`; });
  if (rest > 0) open.at(-1)!.section.style.marginBottom = `${rest}px`;
}

/**
 * A column of drawer sections sharing its height. A folded section is its header; the open ones share the rest
 * (`shareHeight`), so with nothing overflowing the headers folded under the last open section sit at the bottom,
 * and with every section folded the headers stack from the top.
 */
export function SectionStack({ children, className }: { children: ReactNode; className?: string }) {
  const stack = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const node = stack.current!;
    const update = () => layout(node);
    const sizes = new ResizeObserver(update);
    const sections = new MutationObserver(update);
    const observe = () => {
      sizes.disconnect();
      sections.disconnect();
      sizes.observe(node);
      for (const section of node.children) {
        sections.observe(section, { childList: true, attributes: true, attributeFilter: ['data-open'] });
        for (const part of section.children) {
          const body = part instanceof HTMLElement && part.dataset.sectionBody !== undefined;
          const measured = body ? part.firstElementChild : part;
          if (measured) sizes.observe(measured);
        }
      }
    };
    const stackChildren = new MutationObserver(() => { observe(); update(); });
    stackChildren.observe(node, { childList: true });
    observe();
    update();
    return () => { sizes.disconnect(); sections.disconnect(); stackChildren.disconnect(); };
  }, []);
  return <div ref={stack} data-section-stack className={clsx('flex h-full min-h-0 flex-col overflow-hidden', className)}>{children}</div>;
}
