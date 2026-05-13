from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common_schema import ServiceError


@dataclass(slots=True)
class GraphNode:
    id: str
    label: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdge:
    source: str
    target: str
    type: str


@dataclass(slots=True)
class GraphPath:
    nodes: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphQueryResponse:
    success: bool
    entities: list[GraphNode] = field(default_factory=list)
    relations: list[GraphEdge] = field(default_factory=list)
    paths: list[GraphPath] = field(default_factory=list)
    explanation: str | None = None
    error: ServiceError | None = None
