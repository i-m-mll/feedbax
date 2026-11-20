import clsx from 'clsx';
import {
  forwardRef, useCallback, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState,
  type KeyboardEvent, type MouseEvent, type ReactNode,
} from 'react';
import { InlineEdit } from './InlineEdit';
import { ListRow } from './ListRow';
import { ContextMenu, ContextMenuContent, ContextMenuTrigger, openContextMenu, opensContextMenu } from './Menu';

export interface TreeNode {
  id: string;
  name: string;
  glyph?: ReactNode;
  trailing?: ReactNode;
  children?: readonly TreeNode[];
  disabled?: boolean;
  /** Drawn in the muted text colour: an element outside what the view is showing. */
  muted?: boolean;
  /** Read after the row's name by assistive technology: a fact the row draws by its tone, such as a count of related runs. */
  description?: string;
  /** False for a row whose name is not the user's, when the tree renames others. */
  renamable?: boolean;
}

export interface TreeHandle {
  /** Opens the row's ancestors and scrolls it into view. */
  reveal: (id: string) => void;
  /** Opens the row and its ancestors. */
  open: (id: string) => void;
  focus: () => void;
  /** Puts the row's name into editing, revealing it first. */
  rename: (id: string) => void;
}

export interface TreeProps {
  nodes: readonly TreeNode[];
  selected: readonly string[];
  onSelectedChange: (ids: string[]) => void;
  /** Return false to refuse the name and stay in editing. F2 and a double-click on the name start it. */
  onRename?: (id: string, name: string) => boolean | void;
  /** Goes into the row: Return, or a double-click outside its name. Return also opens a closed parent. */
  onActivate?: (id: string) => void;
  /** What dragging the row carries, by media type; rows without it do not drag. */
  dragData?: (node: TreeNode) => Readonly<Record<string, string>> | undefined;
  /** The row's commands, as menu items. */
  menu?: (node: TreeNode) => ReactNode;
  defaultExpanded?: Iterable<string>;
  renameLabel?: string;
  'aria-label': string;
  className?: string;
}

interface Entry {
  node: TreeNode;
  parent: string | null;
  depth: number;
}

const typeAheadPause = 700;

function index(nodes: readonly TreeNode[]) {
  const entries = new Map<string, Entry>();
  const walk = (list: readonly TreeNode[], parent: string | null, depth: number) => {
    for (const node of list) {
      entries.set(node.id, { node, parent, depth });
      if (node.children) walk(node.children, node.id, depth + 1);
    }
  };
  walk(nodes, null, 0);
  return entries;
}

const renamable = (node: TreeNode) => !node.disabled && node.renamable !== false;

const branch = (node: TreeNode) => (node.children?.length ?? 0) > 0;

function descendants(node: TreeNode): string[] {
  return (node.children ?? []).flatMap((child) => branch(child) ? [child.id, ...descendants(child)] : []);
}

export const Tree = forwardRef<TreeHandle, TreeProps>(function Tree(
  { nodes, selected, onSelectedChange, onRename, onActivate, dragData, menu, defaultExpanded, renameLabel = 'Name', className, ...aria },
  ref,
) {
  const entries = useMemo(() => index(nodes), [nodes]);
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set(defaultExpanded));
  const [focusId, setFocusId] = useState<string | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const [menuNode, setMenuNode] = useState<TreeNode | null>(null);
  const rows = useRef(new Map<string, HTMLDivElement>());
  const pendingFocus = useRef(false);
  const pendingScroll = useRef<string | null>(null);
  const anchor = useRef<string | null>(null);
  const emitted = useRef<string | null>(null);
  const typed = useRef({ text: '', at: 0 });

  const visible = useMemo(() => {
    const list: TreeNode[] = [];
    const walk = (level: readonly TreeNode[]) => {
      for (const node of level) {
        list.push(node);
        if (branch(node) && expanded.has(node.id)) walk(node.children!);
      }
    };
    walk(nodes);
    return list;
  }, [nodes, expanded]);

  const branches = useMemo(() => [...entries.values()].some(({ node }) => branch(node)), [entries]);
  const selectedSet = useMemo(() => new Set(selected), [selected]);
  const holders = useMemo(() => {
    const set = new Set<string>();
    for (const id of selected) {
      for (let parent = entries.get(id)?.parent; parent; parent = entries.get(parent)?.parent ?? null) set.add(parent);
    }
    return set;
  }, [entries, selected]);

  const ancestors = useCallback((id: string) => {
    const list: string[] = [];
    for (let parent = entries.get(id)?.parent; parent; parent = entries.get(parent)?.parent ?? null) list.push(parent);
    return list;
  }, [entries]);

  const reveal = useCallback((id: string) => {
    if (!entries.has(id)) return;
    const closed = ancestors(id).filter((parent) => !expanded.has(parent));
    setFocusId(id);
    if (!closed.length) return rows.current.get(id)?.scrollIntoView({ block: 'nearest' });
    setExpanded((current) => new Set([...current, ...closed]));
    pendingScroll.current = id;
  }, [ancestors, entries, expanded]);

  const visibleIds = useMemo(() => new Set(visible.map((node) => node.id)), [visible]);
  const current = focusId && visibleIds.has(focusId) ? focusId
    : selected.find((id) => visibleIds.has(id)) ?? visible[0]?.id ?? null;

  useImperativeHandle(ref, () => ({
    reveal,
    open: (id) => {
      if (!entries.has(id)) return;
      const ids = [id, ...ancestors(id)];
      setExpanded((now) => ids.every((item) => now.has(item)) ? now : new Set([...now, ...ids]));
    },
    focus: () => { if (current) rows.current.get(current)?.focus(); },
    rename: (id) => {
      const entry = entries.get(id);
      if (!onRename || !entry || !renamable(entry.node)) return;
      reveal(id);
      setEditing(id);
    },
  }), [ancestors, current, entries, onRename, reveal]);

  useLayoutEffect(() => {
    const key = selected.join('\u0000');
    if (key === emitted.current) return;
    emitted.current = key;
    const last = selected[selected.length - 1];
    if (last) {
      anchor.current = last;
      reveal(last);
    }
  }, [reveal, selected]);

  useLayoutEffect(() => {
    const id = pendingScroll.current;
    if (id && visibleIds.has(id)) {
      pendingScroll.current = null;
      rows.current.get(id)?.scrollIntoView({ block: 'nearest' });
    }
    if (pendingFocus.current && current) {
      pendingFocus.current = false;
      rows.current.get(current)?.focus();
    }
  });

  const emit = (ids: string[]) => {
    emitted.current = ids.join('\u0000');
    onSelectedChange(ids);
  };

  const siblings = (id: string) => {
    const parent = entries.get(id)?.parent ?? null;
    return parent === null ? nodes : entries.get(parent)!.node.children!;
  };

  const range = (from: string, to: string) => {
    const level = siblings(from);
    if (entries.get(from)?.parent !== entries.get(to)?.parent) return [to];
    const a = level.findIndex((node) => node.id === from);
    const b = level.findIndex((node) => node.id === to);
    return level.slice(Math.min(a, b), Math.max(a, b) + 1).filter((node) => !node.disabled).map((node) => node.id);
  };

  const move = (id: string, extend = false) => {
    pendingFocus.current = true;
    setFocusId(id);
    if (entries.get(id)?.node.disabled) return;
    if (extend && anchor.current) return emit(range(anchor.current, id));
    anchor.current = id;
    emit([id]);
  };

  const sameGroup = (id: string) => selected.every((other) => entries.get(other)?.parent === entries.get(id)?.parent);

  const toggle = (node: TreeNode, open: boolean, deep: boolean) => {
    const ids = deep ? [node.id, ...descendants(node)] : [node.id];
    setExpanded((now) => {
      const next = new Set(now);
      for (const id of ids) {
        if (open) next.add(id);
        else next.delete(id);
      }
      return next;
    });
  };

  const typeAhead = (key: string, from: number) => {
    const now = performance.now();
    const text = now - typed.current.at > typeAheadPause ? key : typed.current.text + key;
    typed.current = { text, at: now };
    const lower = text.toLowerCase();
    const start = text.length === 1 ? from + 1 : from;
    for (let step = 0; step < visible.length; step += 1) {
      const node = visible[(start + step) % visible.length];
      if (node.name.toLowerCase().startsWith(lower)) return move(node.id);
    }
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!current || editing) return;
    const at = visible.findIndex((node) => node.id === current);
    const node = visible[at];
    const open = expanded.has(node.id);
    const level = siblings(node.id);
    const position = level.findIndex((sibling) => sibling.id === node.id);
    const handled = () => event.preventDefault();
    const row = rows.current.get(node.id);
    if (menu && row && opensContextMenu(event)) {
      handled();
      event.stopPropagation();
      return openContextMenu(row);
    }
    switch (event.key) {
      case 'ArrowDown':
        handled();
        if (event.shiftKey) return position < level.length - 1 && move(level[position + 1].id, true);
        return at < visible.length - 1 && move(visible[at + 1].id);
      case 'ArrowUp':
        handled();
        if (event.shiftKey) return position > 0 && move(level[position - 1].id, true);
        return at > 0 && move(visible[at - 1].id);
      case 'ArrowRight':
        handled();
        if (!branch(node)) return;
        if (!open) return toggle(node, true, event.altKey);
        return move(node.children![0].id);
      case 'ArrowLeft': {
        handled();
        if (branch(node) && open) return toggle(node, false, event.altKey);
        const parent = entries.get(node.id)?.parent;
        return parent && move(parent);
      }
      case 'Home':
        handled();
        return move(visible[0].id);
      case 'End':
        handled();
        return move(visible[visible.length - 1].id);
      case 'Enter':
        handled();
        if (branch(node) && !open) toggle(node, true, false);
        if (!node.disabled) onActivate?.(node.id);
        return;
      case 'F2':
        handled();
        if (onRename && renamable(node)) setEditing(node.id);
        return;
      case ' ':
        handled();
        if (event.metaKey || event.ctrlKey) {
          if (!sameGroup(node.id)) return move(node.id);
          return emit(selectedSet.has(node.id) ? selected.filter((id) => id !== node.id) : [...selected, node.id]);
        }
        return move(node.id);
      default:
        if (event.key.length === 1 && !event.metaKey && !event.ctrlKey && !event.altKey) {
          handled();
          typeAhead(event.key, at);
        }
    }
  };

  const onRowClick = (event: MouseEvent, node: TreeNode) => {
    setFocusId(node.id);
    if (node.disabled) return;
    if (event.shiftKey && anchor.current) return emit(range(anchor.current, node.id));
    if ((event.metaKey || event.ctrlKey) && sameGroup(node.id)) {
      anchor.current = node.id;
      return emit(selectedSet.has(node.id) ? selected.filter((id) => id !== node.id) : [...selected, node.id]);
    }
    anchor.current = node.id;
    emit([node.id]);
  };

  const onContextMenu = (node: TreeNode) => {
    setMenuNode(node);
    setFocusId(node.id);
    if (!selectedSet.has(node.id) && !node.disabled) {
      anchor.current = node.id;
      emit([node.id]);
    }
  };

  const tree = (
    <div
      role="tree"
      aria-multiselectable
      onKeyDown={onKeyDown}
      className={clsx('grid min-w-0 content-start gap-px p-1', className)}
      {...aria}
    >
      {visible.map((node) => {
        const { depth, parent } = entries.get(node.id)!;
        const level = siblings(node.id);
        const open = branch(node) ? expanded.has(node.id) : branches ? null : undefined;
        const carried = node.disabled ? undefined : dragData?.(node);
        return (
          <ListRow
            key={node.id}
            ref={(element) => {
              if (element) rows.current.set(node.id, element);
              else rows.current.delete(node.id);
            }}
            role="treeitem"
            aria-level={depth + 1}
            aria-setsize={level.length}
            aria-posinset={level.findIndex((sibling) => sibling.id === node.id) + 1}
            data-parent={parent ?? undefined}
            tabIndex={node.id === current ? 0 : -1}
            depth={depth}
            expanded={open}
            onExpandedChange={(next, deep) => toggle(node, next, deep)}
            glyph={node.glyph}
            trailing={node.trailing}
            selected={selectedSet.has(node.id)}
            holdsSelection={open === false && holders.has(node.id)}
            disabled={node.disabled}
            className={node.muted ? 'text-shell-muted' : undefined}
            aria-description={node.description}
            name={onRename && renamable(node) ? <span data-tree-name>
              <InlineEdit
                value={node.name}
                editing={editing === node.id}
                onEditingChange={(next) => {
                  setEditing(next ? node.id : null);
                  if (!next) pendingFocus.current = true;
                }}
                onCommit={(name) => onRename(node.id, name)}
                aria-label={renameLabel}
              />
            </span> : node.name}
            onClick={(event) => onRowClick(event, node)}
            onDoubleClick={(event) => {
              if (node.disabled) return;
              if (onRename && renamable(node) && event.target instanceof Element && event.target.closest('[data-tree-name]')) setEditing(node.id);
              else onActivate?.(node.id);
            }}
            onContextMenu={() => onContextMenu(node)}
            draggable={carried ? true : undefined}
            onDragStart={carried && ((event) => {
              for (const [type, data] of Object.entries(carried)) event.dataTransfer.setData(type, data);
            })}
          />
        );
      })}
    </div>
  );

  if (!menu) return tree;
  return (
    <ContextMenu>
      <ContextMenuTrigger render={tree} />
      <ContextMenuContent>{menuNode ? menu(menuNode) : null}</ContextMenuContent>
    </ContextMenu>
  );
});
