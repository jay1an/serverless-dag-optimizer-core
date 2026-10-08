"""Workflow DAG definitions and YAML loading.

This module is intentionally small and parameter-free. It owns only the
structural workflow definition: nodes, actions, parent edges, and common DAG
queries such as topological order, sources, sinks, and descendants.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


@dataclass(frozen=True)
class WorkflowNode:
    """One node/stage in a workflow DAG."""

    name: str
    action: str
    parents: tuple[str, ...]


# Backward-friendly alias for the old project naming.
NodeSpec = WorkflowNode


@dataclass(frozen=True)
class WorkflowSpec:
    """A loaded workflow DAG."""

    workflow_name: str
    namespace: str
    entry: str
    nodes: dict[str, WorkflowNode]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate entry, parent references, and acyclicity."""

        if self.entry not in self.nodes:
            raise ValueError(f"entry node {self.entry!r} is not defined in nodes")

        missing_parents: dict[str, list[str]] = {}
        for node in self.nodes.values():
            missing = [parent for parent in node.parents if parent not in self.nodes]
            if missing:
                missing_parents[node.name] = missing
        if missing_parents:
            raise ValueError(f"unknown parent nodes: {missing_parents}")

        # Topological sort raises if a cycle exists.
        self.topological_order()

    def parents_of(self, node_name: str) -> tuple[str, ...]:
        self._require_node(node_name)
        return tuple(self.nodes[node_name].parents)

    def children_of(self, node_name: str) -> tuple[str, ...]:
        self._require_node(node_name)
        return tuple(
            node.name for node in self.nodes.values() if node_name in node.parents
        )

    def sources(self) -> tuple[str, ...]:
        return tuple(node.name for node in self.nodes.values() if not node.parents)

    def sinks(self) -> tuple[str, ...]:
        return tuple(name for name in self.nodes if not self.children_of(name))

    def topological_order(self) -> tuple[str, ...]:
        """Return node names in topological order.

        The order is stable with respect to the YAML node order where multiple
        nodes are simultaneously ready.
        """

        remaining = set(self.nodes)
        emitted: list[str] = []
        emitted_set: set[str] = set()

        while remaining:
            ready = [
                name
                for name in self.nodes
                if name in remaining
                and all(parent in emitted_set for parent in self.nodes[name].parents)
            ]
            if not ready:
                cycle_nodes = ", ".join(sorted(remaining))
                raise ValueError(f"workflow graph contains a cycle involving: {cycle_nodes}")
            for name in ready:
                remaining.remove(name)
                emitted.append(name)
                emitted_set.add(name)
        return tuple(emitted)

    def descendants_of(self, node_name: str) -> tuple[str, ...]:
        """Return all transitive children of ``node_name`` in topological order."""

        self._require_node(node_name)
        descendants: set[str] = set()
        stack = list(self.children_of(node_name))
        while stack:
            child = stack.pop()
            if child in descendants:
                continue
            descendants.add(child)
            stack.extend(self.children_of(child))
        return tuple(name for name in self.topological_order() if name in descendants)

    def ready_nodes(
        self, completed: Iterable[str], running: Iterable[str]
    ) -> tuple[WorkflowNode, ...]:
        completed_set = set(completed)
        running_set = set(running)
        ready: list[WorkflowNode] = []
        for node in self.nodes.values():
            if node.name in completed_set or node.name in running_set:
                continue
            if all(parent in completed_set for parent in node.parents):
                ready.append(node)
        return tuple(ready)

    def _require_node(self, node_name: str) -> None:
        if node_name not in self.nodes:
            raise KeyError(f"unknown workflow node {node_name!r}")


def load_workflow(path: str | Path) -> WorkflowSpec:
    """Load a workflow YAML file."""

    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, Mapping):
        raise ValueError(f"workflow file must contain a mapping: {path}")
    _reject_unknown_keys(
        raw,
        allowed={"workflow_name", "namespace", "entry", "nodes"},
        context=f"workflow file {path}",
    )

    raw_nodes = raw.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError(f"workflow file must contain a list field 'nodes': {path}")

    nodes: dict[str, WorkflowNode] = {}
    for item in raw_nodes:
        if not isinstance(item, Mapping):
            raise ValueError(f"workflow node must be a mapping, got {item!r}")
        _reject_unknown_keys(
            item,
            allowed={"name", "action", "parents"},
            context=f"workflow node in {path}",
        )
        name = str(_required(item, "name"))
        if name in nodes:
            raise ValueError(f"duplicate workflow node {name!r}")
        node = WorkflowNode(
            name=name,
            action=str(_required(item, "action")),
            parents=tuple(str(parent) for parent in item.get("parents", [])),
        )
        nodes[node.name] = node

    return WorkflowSpec(
        workflow_name=str(_required(raw, "workflow_name")),
        namespace=str(raw.get("namespace", "guest")),
        entry=str(_required(raw, "entry")),
        nodes=nodes,
    )


def suffix_action_name(action: str, suffix: str) -> str:
    """Append a memory-tier suffix to an OpenWhisk action name."""

    if not suffix:
        return action
    if "/" not in action:
        return f"{action}{suffix}"
    prefix, name = action.rsplit("/", 1)
    return f"{prefix}/{name}{suffix}"


def with_action_suffix(workflow: WorkflowSpec, suffix: str) -> WorkflowSpec:
    """Return a copy of ``workflow`` with every action name suffixed."""

    if not suffix:
        return workflow
    return WorkflowSpec(
        workflow_name=workflow.workflow_name,
        namespace=workflow.namespace,
        entry=workflow.entry,
        nodes={
            name: replace(node, action=suffix_action_name(node.action, suffix))
            for name, node in workflow.nodes.items()
        },
    )


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required workflow field {key!r}")
    return mapping[key]


def _reject_unknown_keys(
    mapping: Mapping[str, Any], allowed: set[str], context: str
) -> None:
    unknown = sorted(set(mapping).difference(allowed))
    if unknown:
        raise ValueError(
            f"{context} contains deployment/workload fields that do not belong "
            f"in the public DAG schema: {unknown}"
        )
