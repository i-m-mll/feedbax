"""Shared graph utilities.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections import deque
from collections.abc import Iterable, Mapping


def topological_sort(adjacency: Mapping[str, Iterable[str]]) -> list[str]:
    """Return a topological ordering of a directed acyclic graph.

    If cycles remain, the remaining nodes are appended in input order.
    """
    nodes = list(adjacency.keys())
    indegree = {node: 0 for node in nodes}

    for src, targets in adjacency.items():
        for tgt in targets:
            if tgt not in indegree:
                indegree[tgt] = 0
                nodes.append(tgt)
            indegree[tgt] += 1

    queue = deque([node for node in nodes if indegree[node] == 0])
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for tgt in adjacency.get(node, ()):  # allow missing keys
            indegree[tgt] -= 1
            if indegree[tgt] == 0:
                queue.append(tgt)

    if len(order) < len(nodes):
        remaining = [node for node in nodes if node not in order]
        order.extend(remaining)

    return order


def detect_cycles_and_sort(
    adjacency: Mapping[str, Iterable[str]],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Detect cycle-forming edges and return a topological order.

    Uses DFS to identify back edges. The remaining acyclic graph is then
    topologically sorted.
    """
    nodes = list(adjacency.keys())
    visited: dict[str, int] = {node: 0 for node in nodes}  # 0=unseen,1=visiting,2=done
    back_edges: list[tuple[str, str]] = []

    def dfs(node: str) -> None:
        visited[node] = 1
        for tgt in adjacency.get(node, ()):  # allow missing keys
            if tgt not in visited:
                visited[tgt] = 0
            if visited[tgt] == 0:
                dfs(tgt)
            elif visited[tgt] == 1:
                back_edges.append((node, tgt))
        visited[node] = 2

    for node in nodes:
        if visited[node] == 0:
            dfs(node)

    acyclic_adj = {node: set(adjacency.get(node, ())) for node in nodes}
    for src, tgt in back_edges:
        if src in acyclic_adj:
            acyclic_adj[src].discard(tgt)

    order = topological_sort(acyclic_adj)
    return order, back_edges

