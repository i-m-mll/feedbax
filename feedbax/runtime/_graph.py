"""Shared graph utilities.

:copyright: Copyright 2024 by MLL <mll@mll.bio>.
:license: Apache 2.0. See LICENSE for details.
"""

from collections.abc import Iterable, Mapping
from heapq import heapify, heappop, heappush


def topological_sort(adjacency: Mapping[str, Iterable[str]]) -> list[str]:
    """Return the topological order with lexical priority among ready node IDs.

    If cycles remain, the remaining nodes are appended in lexical order.
    """
    adjacency = _canonical_adjacency(adjacency)
    indegree = {node: 0 for node in adjacency}

    for targets in adjacency.values():
        for tgt in targets:
            indegree[tgt] += 1

    queue = [node for node in adjacency if indegree[node] == 0]
    heapify(queue)
    order: list[str] = []

    while queue:
        node = heappop(queue)
        order.append(node)
        for tgt in adjacency[node]:
            indegree[tgt] -= 1
            if indegree[tgt] == 0:
                heappush(queue, tgt)

    if len(order) < len(adjacency):
        completed = set(order)
        order.extend(node for node in adjacency if node not in completed)

    return order


def _canonical_adjacency(adjacency: Mapping[str, Iterable[str]]) -> dict[str, tuple[str, ...]]:
    targets = {node: tuple(sorted(set(edges))) for node, edges in adjacency.items()}
    nodes = set(targets).union(*(set(edges) for edges in targets.values()))
    return {node: targets.get(node, ()) for node in sorted(nodes)}


def detect_cycles_and_sort(
    adjacency: Mapping[str, Iterable[str]],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Detect cycle-forming edges and return a topological order.

    Uses lexical DFS traversal to identify back edges deterministically. The
    remaining acyclic graph uses lexical priority among ready node IDs.
    """
    adjacency = _canonical_adjacency(adjacency)
    nodes = tuple(adjacency)
    visited: dict[str, int] = {node: 0 for node in nodes}  # 0=unseen,1=visiting,2=done
    back_edges: list[tuple[str, str]] = []

    def dfs(node: str) -> None:
        visited[node] = 1
        for tgt in adjacency[node]:
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

