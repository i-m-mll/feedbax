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
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}

const header: Record<SectionLevel, string> = {
  drawer: 'border-b border-shell-divider text-sm font-medium text-shell-muted',
  group: 'text-xs font-medium text-shell-muted',
  field: 'text-xs text-shell-muted',
};

export function Section({
  name, nameId, level = 'drawer', actions, lead, open: controlled, defaultOpen = true, onOpenChange, children, className, bodyClassName,
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
      <div className={clsx('flex h-row shrink-0 items-center gap-0.5', level !== 'field' && 'px-1', header[level], glyph)}>
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
      {lead && open ? <div className="shrink-0">{lead}</div> : null}
      {level === 'drawer'
        ? <ScrollFade id={body} hidden={!open} data-section-body className="flex-none">{content}</ScrollFade>
        : <div id={body} hidden={!open}>{content}</div>}
    </section>
  );
}

/**
 * Heights for the open sections of a stack, given each one's content height and the height they share: each takes an equal
 * share, one whose content is shorter takes only its content and gives the rest to those that overflow, and what nothing
 * needs goes to the last.
 */
export function shareHeight(contents: readonly number[], available: number): number[] {
  const heights = contents.map(() => 0);
  let remaining = Math.max(0, available);
  let active = contents.map((_, index) => index);
  while (active.length) {
    const share = remaining / active.length;
    const fitting = active.filter((index) => contents[index] <= share);
    if (!fitting.length) {
      for (const index of active) heights[index] = share;
      return heights;
    }
    for (const index of fitting) {
      heights[index] = contents[index];
      remaining -= contents[index];
    }
    active = active.filter((index) => contents[index] > share);
  }
  if (heights.length) heights[heights.length - 1] += remaining;
  return heights;
}

function layout(stack: HTMLElement) {
  const open: { body: HTMLElement; content: number }[] = [];
  let fixed = 0;
  for (const section of stack.children) {
    if (!(section instanceof HTMLElement) || section.dataset.section !== 'drawer') continue;
    const body = section.querySelector<HTMLElement>(':scope > [data-section-body]');
    if (!body) continue;
    fixed += section.offsetHeight - body.offsetHeight;
    if (section.dataset.open === undefined) body.style.removeProperty('height');
    else open.push({ body, content: (body.firstElementChild as HTMLElement | null)?.offsetHeight ?? 0 });
  }
  const heights = shareHeight(open.map(({ content }) => content), stack.clientHeight - fixed);
  open.forEach(({ body }, index) => { body.style.height = `${heights[index]}px`; });
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
