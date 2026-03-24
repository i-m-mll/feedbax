/**
 * StateFieldTree — hierarchical tree of state fields with connectable handles.
 *
 * Renders a collapsible, indented tree where each node (branch or leaf) carries
 * a React Flow Handle on its right edge. The handle ID is the full dot-path
 * (e.g. "states.net.hidden") so wires encode which specific data flows through.
 *
 * Branch nodes show an expand/collapse chevron. All handles are type "source"
 * (output), positioned on the right side of the DataSourceNode.
 */

import { useCallback, useState, type CSSProperties } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { StateFieldNode } from '@/types/analysis';
import { ChevronRight, ChevronDown } from 'lucide-react';
import clsx from 'clsx';

/** Height of each row in the tree (px). */
export const FIELD_ROW_HEIGHT = 20;
/** Indentation per depth level (px). */
const INDENT_PX = 14;
/** Handle diameter (px). */
const HANDLE_SIZE = 7;

// ---------------------------------------------------------------------------
// Flatten helpers
// ---------------------------------------------------------------------------

interface FlatEntry {
  node: StateFieldNode;
  depth: number;
  isLeaf: boolean;
}

function flattenVisible(
  nodes: StateFieldNode[],
  expandedPaths: ReadonlySet<string>,
  depth: number = 0,
): FlatEntry[] {
  const result: FlatEntry[] = [];
  for (const node of nodes) {
    const isLeaf = !node.children || node.children.length === 0;
    result.push({ node, depth, isLeaf });
    if (node.children && expandedPaths.has(node.path)) {
      result.push(...flattenVisible(node.children, expandedPaths, depth + 1));
    }
  }
  return result;
}

/** Count total visible rows given current expansion state. */
export function countVisibleRows(
  nodes: StateFieldNode[],
  expandedPaths: ReadonlySet<string>,
): number {
  let count = 0;
  for (const node of nodes) {
    count += 1;
    if (node.children && expandedPaths.has(node.path)) {
      count += countVisibleRows(node.children, expandedPaths);
    }
  }
  return count;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface StateFieldTreeProps {
  /** Tree nodes to render. */
  nodes: StateFieldNode[];
  /** Set of expanded paths (controlled from parent). */
  expandedPaths: ReadonlySet<string>;
  /** Called when a branch chevron is toggled. */
  onToggle: (path: string) => void;
  /** Vertical padding offset from top of the body div (px). */
  bodyPadding: number;
}

export function StateFieldTree({
  nodes,
  expandedPaths,
  onToggle,
  bodyPadding,
}: StateFieldTreeProps) {
  const entries = flattenVisible(nodes, expandedPaths);

  return (
    <>
      {entries.map((entry, index) => {
        const { node, depth, isLeaf } = entry;
        const hasChildren = !isLeaf;
        const isExpanded = expandedPaths.has(node.path);
        const top = bodyPadding + FIELD_ROW_HEIGHT * (index + 0.5);

        return (
          <div key={node.path}>
            {/* Connectable handle — present on every tree node */}
            <Handle
              type="source"
              position={Position.Right}
              id={node.path}
              style={{
                top,
                right: -6,
                transform: 'translateY(-50%)',
                width: `${HANDLE_SIZE}px`,
                height: `${HANDLE_SIZE}px`,
              } satisfies CSSProperties}
              className={clsx(
                'border border-white shadow-soft',
                isLeaf ? 'bg-slate-400' : 'bg-slate-300',
              )}
            />

            {/* Label row */}
            <div
              className="absolute flex items-center text-[11px] select-none"
              style={{
                top,
                left: depth * INDENT_PX + 4,
                right: 18,
                transform: 'translateY(-50%)',
                height: FIELD_ROW_HEIGHT,
              }}
            >
              {hasChildren ? (
                <button
                  className="flex items-center gap-0.5 text-slate-500 hover:text-slate-700 cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggle(node.path);
                  }}
                >
                  {isExpanded ? (
                    <ChevronDown className="w-3 h-3 shrink-0" />
                  ) : (
                    <ChevronRight className="w-3 h-3 shrink-0" />
                  )}
                  <span className="text-slate-500 font-medium">{node.label}</span>
                </button>
              ) : (
                <span className="ml-3.5 text-slate-400">{node.label}</span>
              )}
            </div>
          </div>
        );
      })}
    </>
  );
}

// ---------------------------------------------------------------------------
// Hook — manage expansion state & expose visible row count
// ---------------------------------------------------------------------------

export function useFieldTreeExpansion(nodes: StateFieldNode[]) {
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());

  const toggleExpand = useCallback((path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);

  const visibleCount = countVisibleRows(nodes, expandedPaths);

  return { expandedPaths, toggleExpand, visibleCount };
}
