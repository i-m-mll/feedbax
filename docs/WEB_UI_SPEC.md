# Feedbax Web UI Specification

> Canvas-based interface for constructing, visualizing, and training neural control models

**Version**: 0.1.0 (Draft)
**Date**: 2026-01-26
**Author**: MLL <mll@mll.bio>, Claude Opus 4.5

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Technology Stack](#3-technology-stack)
4. [Data Model & Serialization](#4-data-model--serialization)
5. [Frontend Architecture](#5-frontend-architecture)
6. [Backend Architecture](#6-backend-architecture)
7. [UI/UX Design](#7-uiux-design)
8. [Canvas & Graph Editing](#8-canvas--graph-editing)
9. [Component System](#9-component-system)
10. [Training Configuration](#10-training-configuration)
11. [Execution & Visualization](#11-execution--visualization)
12. [State Management](#12-state-management)
13. [API Specification](#13-api-specification)
14. [File Structure](#14-file-structure)
15. [Implementation Phases](#15-implementation-phases)
16. [Future Considerations](#16-future-considerations)

---

## 1. Executive Summary

### 1.1 Purpose

Build a modern web application for interactively constructing and training Feedbax models. Users should be able to:

- **Design** model architectures by connecting components on a visual canvas
- **Configure** component parameters, training settings, and loss functions
- **Execute** simulations and training runs with real-time feedback
- **Inspect** model state, data flow, and training progress
- **Export** configurations for offline execution or sharing

### 1.2 Design Philosophy

- **Explicit over implicit**: The canvas is a direct representation of the underlying `Graph` data structure
- **Clean and minimal**: Inspired by modern tools like Claude Code, Linear, Figma
- **Bidirectional**: Changes in UI ↔ JSON ↔ Python objects seamlessly
- **Composable**: Components can be nested (graphs within graphs)
- **Non-destructive**: Operations are reversible; state is versioned

### 1.3 Relationship to Existing Code

The UI is a visual interface to the existing `feedbax.graph` module:

| Python Concept | UI Representation |
|----------------|-------------------|
| `Component` | Node on canvas |
| `Wire` | Edge connecting ports |
| `Graph` | Canvas view (nestable) |
| `input_ports` / `output_ports` | Connection handles on nodes |
| `StateIndex` | Inspectable state in debug panel |
| Graph surgery methods | Drag-drop, context menus |

---

## 2. Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Browser)                        │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │  React Flow   │  │  Zustand      │  │  WebSocket Client │   │
│  │  (Canvas)     │  │  (State)      │  │  (Real-time)      │   │
│  └───────────────┘  └───────────────┘  └───────────────────┘   │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │   REST Client     │                        │
│                    └─────────┬─────────┘                        │
└──────────────────────────────┼──────────────────────────────────┘
                               │ HTTP/WS
┌──────────────────────────────┼──────────────────────────────────┐
│                        Backend (Python)                          │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────┐   │
│  │   FastAPI     │  │  WebSocket    │  │  Background Tasks │   │
│  │   (REST)      │  │  (Streaming)  │  │  (Training)       │   │
│  └───────────────┘  └───────────────┘  └───────────────────┘   │
│                              │                                   │
│                    ┌─────────┴─────────┐                        │
│                    │   Feedbax Core    │                        │
│                    │   (JAX/Equinox)   │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Communication Patterns

| Operation | Protocol | Pattern |
|-----------|----------|---------|
| Load/Save graphs | REST | Request-Response |
| Component CRUD | REST | Request-Response |
| Graph validation | REST | Request-Response |
| Training execution | WebSocket | Streaming (progress, metrics) |
| Simulation preview | WebSocket | Streaming (state snapshots) |
| State inspection | REST | Request-Response |

---

## 3. Technology Stack

### 3.1 Frontend

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Framework | **React 18+** | Mature ecosystem, excellent library support |
| Canvas | **React Flow 12+** | Best-in-class node editor, MIT license, xyflow team |
| Build | **Vite 5+** | Fast HMR, modern defaults, excellent DX |
| Language | **TypeScript 5+** | Type safety, better tooling, self-documenting |
| Styling | **Tailwind CSS 3+** | Utility-first, rapid prototyping, consistent design |
| State | **Zustand 4+** | Lightweight, simple API, React Flow compatible |
| HTTP | **TanStack Query 5+** | Caching, background refetch, optimistic updates |
| WebSocket | **Native + custom hooks** | Simple needs, no library overhead |
| Icons | **Lucide React** | Clean, consistent, MIT license |
| UI Components | **Radix UI** | Accessible primitives, unstyled (Tailwind friendly) |

### 3.2 Backend

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Framework | **FastAPI 0.100+** | Async, auto-docs, Pydantic integration |
| WebSocket | **FastAPI WebSocket** | Built-in, good enough for our needs |
| Validation | **Pydantic 2+** | Data validation, serialization, OpenAPI |
| Background | **asyncio + threading** | JAX runs in thread pool, async coordination |
| Storage | **Filesystem + SQLite** | Simple, portable, JSON files for graphs |

### 3.3 Development

| Tool | Purpose |
|------|---------|
| **pnpm** | Package management (faster, stricter than npm) |
| **ESLint + Prettier** | Linting, formatting |
| **Vitest** | Unit testing |
| **Playwright** | E2E testing |
| **uv** | Python package management |
| **Ruff** | Python linting/formatting |
| **pytest** | Python testing |

### 3.4 Framework Replaceability Note

React permeates the UI layer deeply (components, hooks, JSX). However, the architecture isolates framework-specific code:

- **Framework-agnostic** (~50% of frontend code): Graph operations, validation logic, serialization, API clients, type definitions
- **Framework-specific** (~50%): React components, hooks, React Flow integration

A future migration to Svelte would require rewriting UI components but preserve all business logic. This is acceptable—the stack is stable for 5+ years.

---

## 4. Data Model & Serialization

### 4.1 Core Types (TypeScript)

```typescript
// Mirrors feedbax.graph.Component
interface ComponentSpec {
  type: string;                          // e.g., "SimpleStagedNetwork", "Mechanics"
  params: Record<string, ParamValue>;    // Component-specific parameters
  input_ports: string[];                 // e.g., ["target", "feedback"]
  output_ports: string[];                // e.g., ["output", "hidden"]
}

// Mirrors feedbax.graph.Wire
interface WireSpec {
  source_node: string;
  source_port: string;
  target_node: string;
  target_port: string;
}

// Mirrors feedbax.graph.Graph
interface GraphSpec {
  nodes: Record<string, ComponentSpec>;
  wires: WireSpec[];
  input_ports: string[];
  output_ports: string[];
  input_bindings: Record<string, [string, string]>;   // ext_port -> [node, port]
  output_bindings: Record<string, [string, string]>;
  metadata?: GraphMetadata;
}

interface GraphMetadata {
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
  version: string;
  author?: string;
  tags?: string[];
}

// UI-specific extensions (not sent to backend)
interface NodeUIState {
  position: { x: number; y: number };
  collapsed: boolean;
  selected: boolean;
}

interface GraphUIState {
  viewport: { x: number; y: number; zoom: number };
  node_states: Record<string, NodeUIState>;
}
```

### 4.2 Parameter Types

```typescript
type ParamValue =
  | number
  | string
  | boolean
  | number[]
  | ParamValue[]
  | Record<string, ParamValue>
  | null;

interface ParamSchema {
  name: string;
  type: "int" | "float" | "bool" | "str" | "enum" | "array" | "object";
  default?: ParamValue;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];              // For enum type
  description?: string;
  required: boolean;
  nested_schema?: ParamSchema[];   // For object type
}
```

### 4.3 Training Configuration Types

```typescript
interface OptimizerSpec {
  type: string;                    // "adam", "sgd", "adamw", etc.
  params: Record<string, ParamValue>;
}

interface LossTermSpec {
  type: string;                    // Loss class name
  weight: number;
  params: Record<string, ParamValue>;
  children?: Record<string, LossTermSpec>;  // For hierarchical TermTree
}

interface TrainingSpec {
  optimizer: OptimizerSpec;
  loss: LossTermSpec;
  n_batches: number;
  batch_size: number;
  n_epochs?: number;
  checkpoint_interval?: number;
  early_stopping?: {
    metric: string;
    patience: number;
    min_delta: number;
  };
}

interface TaskSpec {
  type: string;
  params: Record<string, ParamValue>;
  timeline?: TimelineSpec;
}

interface TimelineSpec {
  epochs: Record<string, [number, number]>;  // name -> [start, end]
  events?: Record<string, number>;           // name -> time
}
```

### 4.4 JSON Serialization Format

Complete project file format:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "Reaching Task Model",
    "description": "Two-link arm reaching to targets",
    "created_at": "2026-01-26T12:00:00Z",
    "updated_at": "2026-01-26T14:30:00Z"
  },
  "graph": {
    "nodes": {
      "network": {
        "type": "SimpleStagedNetwork",
        "params": {
          "hidden_size": 100,
          "input_size": 6,
          "output_size": 2,
          "hidden_type": "GRUCell",
          "out_nonlinearity": "tanh"
        },
        "input_ports": ["target", "feedback"],
        "output_ports": ["output", "hidden"]
      },
      "mechanics": {
        "type": "Mechanics",
        "params": {
          "plant_type": "TwoLinkArm",
          "dt": 0.01
        },
        "input_ports": ["force"],
        "output_ports": ["effector", "state"]
      },
      "feedback": {
        "type": "FeedbackChannel",
        "params": {
          "delay": 5,
          "noise_std": 0.01
        },
        "input_ports": ["input"],
        "output_ports": ["output"]
      }
    },
    "wires": [
      {"source_node": "feedback", "source_port": "output", "target_node": "network", "target_port": "feedback"},
      {"source_node": "network", "source_port": "output", "target_node": "mechanics", "target_port": "force"},
      {"source_node": "mechanics", "source_port": "effector", "target_node": "feedback", "target_port": "input"}
    ],
    "input_ports": ["target"],
    "output_ports": ["effector"],
    "input_bindings": {"target": ["network", "target"]},
    "output_bindings": {"effector": ["mechanics", "effector"]}
  },
  "task": {
    "type": "ReachingTask",
    "params": {
      "n_targets": 8,
      "target_radius": 0.02
    },
    "timeline": {
      "epochs": {
        "pre_movement": [0, 50],
        "movement": [50, 150],
        "hold": [150, 200]
      }
    }
  },
  "training": {
    "optimizer": {
      "type": "adam",
      "params": {"learning_rate": 0.001, "b1": 0.9, "b2": 0.999}
    },
    "loss": {
      "type": "Composite",
      "weight": 1.0,
      "children": {
        "position": {"type": "PositionError", "weight": 1.0, "params": {}},
        "effort": {"type": "EffortCost", "weight": 0.01, "params": {}}
      }
    },
    "n_batches": 1000,
    "batch_size": 64
  },
  "ui_state": {
    "viewport": {"x": 0, "y": 0, "zoom": 1},
    "node_states": {
      "network": {"position": {"x": 300, "y": 200}, "collapsed": false},
      "mechanics": {"position": {"x": 600, "y": 200}, "collapsed": false},
      "feedback": {"position": {"x": 450, "y": 400}, "collapsed": false}
    }
  }
}
```

### 4.5 Python Serialization

```python
# feedbax/web/serialization.py

from pydantic import BaseModel
from feedbax.graph import Graph, Component, Wire

class WireSpec(BaseModel):
    source_node: str
    source_port: str
    target_node: str
    target_port: str

class ComponentSpec(BaseModel):
    type: str
    params: dict
    input_ports: list[str]
    output_ports: list[str]

class GraphSpec(BaseModel):
    nodes: dict[str, ComponentSpec]
    wires: list[WireSpec]
    input_ports: list[str]
    output_ports: list[str]
    input_bindings: dict[str, tuple[str, str]]
    output_bindings: dict[str, tuple[str, str]]

def graph_to_spec(graph: Graph) -> GraphSpec:
    """Serialize a Graph to a GraphSpec for JSON export."""
    ...

def spec_to_graph(spec: GraphSpec, component_registry: dict) -> Graph:
    """Instantiate a Graph from a GraphSpec."""
    ...
```

---

## 5. Frontend Architecture

### 5.1 Application Structure

```
src/
├── main.tsx                 # Entry point
├── App.tsx                  # Root component, routing
├── index.css                # Global styles, Tailwind directives
│
├── components/              # Reusable UI components
│   ├── ui/                  # Base components (Button, Input, etc.)
│   ├── canvas/              # React Flow related
│   │   ├── Canvas.tsx       # Main canvas wrapper
│   │   ├── CustomNode.tsx   # Generic component node
│   │   ├── PortHandle.tsx   # Input/output port handles
│   │   ├── CustomEdge.tsx   # Wire visualization
│   │   └── MiniMap.tsx      # Navigation minimap
│   ├── panels/              # Side panels
│   │   ├── ComponentLibrary.tsx
│   │   ├── PropertiesPanel.tsx
│   │   ├── TrainingPanel.tsx
│   │   └── InspectorPanel.tsx
│   └── layout/              # Layout components
│       ├── Header.tsx
│       ├── Sidebar.tsx
│       └── StatusBar.tsx
│
├── features/                # Feature modules
│   ├── graph/               # Graph editing logic
│   │   ├── operations.ts    # Add/remove/wire operations
│   │   ├── validation.ts    # Graph validation
│   │   └── serialization.ts # JSON <-> internal types
│   ├── training/            # Training configuration
│   │   ├── optimizer.ts
│   │   └── loss.ts
│   └── execution/           # Simulation/training execution
│       ├── runner.ts
│       └── visualization.ts
│
├── stores/                  # Zustand stores
│   ├── graphStore.ts        # Graph state
│   ├── uiStore.ts           # UI state (panels, selection)
│   ├── trainingStore.ts     # Training config
│   └── executionStore.ts    # Execution state
│
├── api/                     # Backend communication
│   ├── client.ts            # REST client
│   ├── websocket.ts         # WebSocket client
│   └── queries.ts           # TanStack Query hooks
│
├── hooks/                   # Custom React hooks
│   ├── useGraph.ts
│   ├── useTraining.ts
│   └── useExecution.ts
│
├── types/                   # TypeScript types
│   ├── graph.ts
│   ├── components.ts
│   ├── training.ts
│   └── api.ts
│
└── utils/                   # Utility functions
    ├── geometry.ts          # Position calculations
    └── format.ts            # Display formatting
```

### 5.2 Component Hierarchy

```
App
├── Header
│   ├── ProjectName
│   ├── FileMenu (New, Open, Save, Export)
│   └── ViewMenu (Panels, Zoom)
│
├── Main (flex container)
│   ├── Sidebar (left, collapsible)
│   │   ├── ComponentLibrary
│   │   │   ├── SearchInput
│   │   │   ├── CategoryList
│   │   │   └── ComponentCard (draggable)
│   │   └── ProjectExplorer (future)
│   │
│   ├── CanvasArea (center, flex-grow)
│   │   ├── Canvas (React Flow)
│   │   │   ├── CustomNode (per component)
│   │   │   │   ├── NodeHeader
│   │   │   │   ├── PortList (inputs)
│   │   │   │   ├── PortList (outputs)
│   │   │   │   └── QuickParams (optional)
│   │   │   └── CustomEdge (per wire)
│   │   ├── MiniMap
│   │   └── CanvasControls (zoom, fit)
│   │
│   └── RightPanel (right, collapsible, tabbed)
│       ├── PropertiesTab
│       │   ├── NodeProperties (when node selected)
│       │   └── GraphProperties (when nothing selected)
│       ├── TrainingTab
│       │   ├── OptimizerConfig
│       │   ├── LossConfig
│       │   └── TrainingControls
│       └── InspectorTab
│           ├── StateTree
│           └── PortValues
│
└── StatusBar
    ├── ConnectionStatus
    ├── ValidationStatus
    └── ExecutionStatus
```

### 5.3 React Flow Integration

```tsx
// components/canvas/Canvas.tsx

import { ReactFlow, Background, Controls, MiniMap } from '@xyflow/react';
import { useGraphStore } from '@/stores/graphStore';
import { CustomNode } from './CustomNode';
import { CustomEdge } from './CustomEdge';

const nodeTypes = {
  component: CustomNode,
};

const edgeTypes = {
  wire: CustomEdge,
};

export function Canvas() {
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect } = useGraphStore();

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      fitView
      snapToGrid
      snapGrid={[16, 16]}
    >
      <Background variant="dots" gap={16} size={1} />
      <Controls />
      <MiniMap />
    </ReactFlow>
  );
}
```

```tsx
// components/canvas/CustomNode.tsx

import { Handle, Position, NodeProps } from '@xyflow/react';
import { ComponentSpec } from '@/types/graph';

interface CustomNodeData {
  spec: ComponentSpec;
  label: string;
}

export function CustomNode({ data, selected }: NodeProps<CustomNodeData>) {
  const { spec, label } = data;

  return (
    <div className={`
      bg-white rounded-lg border-2 shadow-sm min-w-[180px]
      ${selected ? 'border-blue-500' : 'border-gray-200'}
    `}>
      {/* Header */}
      <div className="px-3 py-2 bg-gray-50 rounded-t-lg border-b">
        <span className="font-medium text-sm">{label}</span>
        <span className="text-xs text-gray-500 ml-2">{spec.type}</span>
      </div>

      {/* Ports */}
      <div className="flex justify-between p-2">
        {/* Input ports (left) */}
        <div className="space-y-2">
          {spec.input_ports.map((port) => (
            <div key={port} className="flex items-center">
              <Handle
                type="target"
                position={Position.Left}
                id={port}
                className="w-3 h-3 bg-blue-500"
              />
              <span className="text-xs ml-2">{port}</span>
            </div>
          ))}
        </div>

        {/* Output ports (right) */}
        <div className="space-y-2">
          {spec.output_ports.map((port) => (
            <div key={port} className="flex items-center">
              <span className="text-xs mr-2">{port}</span>
              <Handle
                type="source"
                position={Position.Right}
                id={port}
                className="w-3 h-3 bg-green-500"
              />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

---

## 6. Backend Architecture

### 6.1 Module Structure

```
feedbax/
├── web/
│   ├── __init__.py
│   ├── app.py               # FastAPI application factory
│   ├── config.py            # Configuration (ports, paths)
│   │
│   ├── api/                  # REST endpoints
│   │   ├── __init__.py
│   │   ├── graphs.py         # Graph CRUD
│   │   ├── components.py     # Component registry
│   │   ├── training.py       # Training management
│   │   └── execution.py      # Simulation execution
│   │
│   ├── ws/                   # WebSocket handlers
│   │   ├── __init__.py
│   │   ├── training.py       # Training progress streaming
│   │   └── simulation.py     # Simulation state streaming
│   │
│   ├── services/             # Business logic
│   │   ├── __init__.py
│   │   ├── graph_service.py  # Graph operations
│   │   ├── component_registry.py
│   │   ├── training_service.py
│   │   └── execution_service.py
│   │
│   ├── models/               # Pydantic models
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── component.py
│   │   └── training.py
│   │
│   └── serialization.py      # Graph <-> JSON conversion
```

### 6.2 FastAPI Application

```python
# feedbax/web/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from feedbax.web.api import graphs, components, training, execution
from feedbax.web.ws import training as ws_training, simulation as ws_simulation

def create_app() -> FastAPI:
    app = FastAPI(
        title="Feedbax Web API",
        version="0.1.0",
        description="API for Feedbax model construction and training"
    )

    # CORS for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Vite dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # REST routes
    app.include_router(graphs.router, prefix="/api/graphs", tags=["graphs"])
    app.include_router(components.router, prefix="/api/components", tags=["components"])
    app.include_router(training.router, prefix="/api/training", tags=["training"])
    app.include_router(execution.router, prefix="/api/execution", tags=["execution"])

    # WebSocket routes
    app.include_router(ws_training.router, prefix="/ws", tags=["websocket"])
    app.include_router(ws_simulation.router, prefix="/ws", tags=["websocket"])

    return app

app = create_app()
```

### 6.3 Component Registry

```python
# feedbax/web/services/component_registry.py

from dataclasses import dataclass
from typing import Type, Callable
from pathlib import Path
import importlib.util

from feedbax.graph import Component
from feedbax.nn import SimpleStagedNetwork
from feedbax.mechanics import Mechanics
from feedbax.channel import FeedbackChannel
# ... other built-in components

@dataclass
class ComponentDefinition:
    """Metadata about a registered component type."""
    name: str
    cls: Type[Component]
    category: str
    description: str
    param_schema: list[dict]  # Parameter definitions
    input_ports: tuple[str, ...]
    output_ports: tuple[str, ...]
    icon: str = "box"  # Lucide icon name


class ComponentRegistry:
    """Registry of available component types."""

    def __init__(self):
        self._components: dict[str, ComponentDefinition] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register all built-in Feedbax components."""
        self.register(
            name="SimpleStagedNetwork",
            cls=SimpleStagedNetwork,
            category="Neural Networks",
            description="Recurrent neural network with encoder/decoder stages",
            param_schema=[
                {"name": "hidden_size", "type": "int", "default": 100, "min": 1, "required": True},
                {"name": "input_size", "type": "int", "default": 6, "min": 1, "required": True},
                {"name": "output_size", "type": "int", "default": 2, "min": 1, "required": True},
                {"name": "hidden_type", "type": "enum", "options": ["GRUCell", "LSTMCell", "Linear"], "default": "GRUCell"},
                {"name": "out_nonlinearity", "type": "enum", "options": ["tanh", "relu", "identity"], "default": "tanh"},
                # ... more params
            ],
        )
        # ... register other built-ins

    def register(self, name: str, cls: Type[Component], **kwargs):
        """Register a component type."""
        self._components[name] = ComponentDefinition(
            name=name,
            cls=cls,
            input_ports=cls.input_ports,
            output_ports=cls.output_ports,
            **kwargs
        )

    def load_user_components(self, path: Path):
        """Load user-defined components from a directory."""
        if not path.exists():
            return

        for py_file in path.glob("*.py"):
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for Component subclasses with @register_component decorator
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and
                    issubclass(obj, Component) and
                    obj is not Component and
                    hasattr(obj, '_feedbax_component_meta')):
                    meta = obj._feedbax_component_meta
                    self.register(name=meta.get('name', name), cls=obj, **meta)

    def get(self, name: str) -> ComponentDefinition | None:
        return self._components.get(name)

    def list_all(self) -> list[ComponentDefinition]:
        return list(self._components.values())

    def list_by_category(self) -> dict[str, list[ComponentDefinition]]:
        by_category: dict[str, list[ComponentDefinition]] = {}
        for comp in self._components.values():
            by_category.setdefault(comp.category, []).append(comp)
        return by_category
```

### 6.4 Training Service

```python
# feedbax/web/services/training_service.py

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator
import threading
import queue

import jax
from feedbax.train import TaskTrainer
from feedbax.graph import Graph

class TrainingStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class TrainingProgress:
    batch: int
    total_batches: int
    loss: float
    metrics: dict
    status: TrainingStatus

class TrainingService:
    """Manages training jobs with progress streaming."""

    def __init__(self):
        self._status = TrainingStatus.IDLE
        self._progress_queue: queue.Queue[TrainingProgress] = queue.Queue()
        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None

    def start_training(
        self,
        graph: Graph,
        training_spec: dict,
        task_spec: dict,
    ):
        """Start a training job in a background thread."""
        if self._status == TrainingStatus.RUNNING:
            raise RuntimeError("Training already in progress")

        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._training_loop,
            args=(graph, training_spec, task_spec),
            daemon=True
        )
        self._thread.start()
        self._status = TrainingStatus.RUNNING

    def _training_loop(self, graph: Graph, training_spec: dict, task_spec: dict):
        """Execute training (runs in separate thread)."""
        try:
            # Build trainer from specs
            trainer = self._build_trainer(graph, training_spec, task_spec)

            for batch_idx in range(training_spec["n_batches"]):
                if self._stop_flag.is_set():
                    break

                # Run one training step
                loss, metrics = trainer.step()

                # Report progress
                self._progress_queue.put(TrainingProgress(
                    batch=batch_idx,
                    total_batches=training_spec["n_batches"],
                    loss=float(loss),
                    metrics=metrics,
                    status=TrainingStatus.RUNNING,
                ))

            self._status = TrainingStatus.COMPLETED
        except Exception as e:
            self._progress_queue.put(TrainingProgress(
                batch=0, total_batches=0, loss=0.0,
                metrics={"error": str(e)},
                status=TrainingStatus.ERROR,
            ))
            self._status = TrainingStatus.ERROR

    async def stream_progress(self) -> AsyncIterator[TrainingProgress]:
        """Async generator for progress updates."""
        while self._status == TrainingStatus.RUNNING:
            try:
                progress = self._progress_queue.get_nowait()
                yield progress
            except queue.Empty:
                await asyncio.sleep(0.1)

        # Drain remaining items
        while not self._progress_queue.empty():
            yield self._progress_queue.get_nowait()

    def stop_training(self):
        """Stop the current training job."""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._status = TrainingStatus.IDLE
```

---

## 7. UI/UX Design

### 7.1 Design Principles

1. **Information density**: Show relevant information without clutter
2. **Progressive disclosure**: Basic view by default, details on demand
3. **Spatial consistency**: Predictable locations for elements
4. **Immediate feedback**: Actions have visible results
5. **Keyboard accessible**: Power users can work without mouse

### 7.2 Color Palette

```css
/* Neutral */
--gray-50: #fafafa;
--gray-100: #f4f4f5;
--gray-200: #e4e4e7;
--gray-300: #d4d4d8;
--gray-400: #a1a1aa;
--gray-500: #71717a;
--gray-600: #52525b;
--gray-700: #3f3f46;
--gray-800: #27272a;
--gray-900: #18181b;

/* Accent (Blue) */
--blue-500: #3b82f6;
--blue-600: #2563eb;

/* Status */
--green-500: #22c55e;  /* Success, output ports */
--red-500: #ef4444;    /* Error */
--yellow-500: #eab308; /* Warning */
--purple-500: #a855f7; /* Cycle wires */
```

### 7.3 Layout Specifications

```
┌─────────────────────────────────────────────────────────────────┐
│ Header (h: 48px)                                                │
│ ┌─────────┬──────────────────────────────────┬───────────────┐ │
│ │ Logo    │ Project: Reaching Model ▼        │ ⚙️  👤        │ │
│ └─────────┴──────────────────────────────────┴───────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Main Area                                                       │
│ ┌────────────┬────────────────────────────┬──────────────────┐ │
│ │ Sidebar    │ Canvas                     │ Right Panel      │ │
│ │ (w: 240px) │ (flex-grow)                │ (w: 320px)       │ │
│ │            │                            │                  │ │
│ │ Components │                            │ Properties       │ │
│ │ ├─ Neural  │   ┌─────────┐              │ ├─ network       │ │
│ │ │  ├─ RNN  │   │ network │──────┐       │ │  type: RNN     │ │
│ │ │  └─ MLP  │   └─────────┘      │       │ │  hidden: 100   │ │
│ │ ├─ Physics │        │           ▼       │ │  ...           │ │
│ │ │  └─ Arm  │        │      ┌────────┐   │ └────────────────│ │
│ │ └─ Signals │        │      │mechanics│   │                  │ │
│ │    └─ Delay│        │      └────────┘   │ Training         │ │
│ │            │        │           │       │ ├─ Optimizer     │ │
│ │            │        ▼           │       │ │  └─ Adam       │ │
│ │            │   ┌─────────┐      │       │ ├─ Loss          │ │
│ │            │   │feedback │◄─────┘       │ │  └─ Position   │ │
│ │            │   └─────────┘              │ └─ [Start]       │ │
│ │            │                            │                  │ │
│ └────────────┴────────────────────────────┴──────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│ Status Bar (h: 24px)                                            │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ ● Connected │ ✓ Valid graph │ Idle                          │ │
│ └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 7.4 Node Design

```
Standard Node (expanded)
┌────────────────────────────────────┐
│ ■ network                SimpleSt… │  <- Header: name + type
├────────────────────────────────────┤
│ ●─ target          output ─●       │  <- Ports with labels
│ ●─ feedback        hidden ─●       │     Blue = input, Green = output
└────────────────────────────────────┘

Standard Node (collapsed)
┌────────────────────────────────────┐
│ ■ network  [2] ─●──── [2] ●        │  <- Compact: port counts
└────────────────────────────────────┘

Selected Node
┌────────────────────────────────────┐  <- Blue border
│ ■ network                SimpleSt… │
├────────────────────────────────────┤
│ ●─ target          output ─●       │
│ ●─ feedback        hidden ─●       │
└────────────────────────────────────┘
  ↑                                ↑
  Drag handles appear on hover
```

### 7.5 Wire Design

```
Regular wire:     ─────────────────────  (gray, smooth bezier)
Selected wire:    ═════════════════════  (blue, thicker)
Cycle wire:       ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  (purple, dashed)
Invalid wire:     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  (red, during drag if incompatible)
```

### 7.6 Interaction Patterns

| Action | Trigger | Result |
|--------|---------|--------|
| Add node | Drag from sidebar | Node appears at drop position |
| Connect ports | Drag from output to input | Wire created |
| Delete node | Select + Delete/Backspace | Node + its wires removed |
| Delete wire | Select + Delete/Backspace | Wire removed |
| Multi-select | Shift+Click or box select | Multiple items selected |
| Pan canvas | Middle-drag or Space+drag | Canvas moves |
| Zoom | Scroll wheel or pinch | Canvas zooms |
| Edit properties | Select node | Properties panel updates |
| Expand/collapse | Double-click node | Node toggles expanded state |
| Insert between | Drop node on existing wire | Old wire split, new node inserted |

### 7.7 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl/Cmd + S` | Save project |
| `Ctrl/Cmd + Z` | Undo |
| `Ctrl/Cmd + Shift + Z` | Redo |
| `Delete` / `Backspace` | Delete selected |
| `Ctrl/Cmd + A` | Select all |
| `Ctrl/Cmd + D` | Duplicate selected |
| `Escape` | Deselect / cancel |
| `Space` (hold) | Pan mode |
| `F` | Fit view to content |
| `1-9` | Zoom to preset levels |
| `Ctrl/Cmd + Enter` | Start/stop training |

---

## 8. Canvas & Graph Editing

### 8.1 Graph Operations (Frontend)

```typescript
// features/graph/operations.ts

import { GraphSpec, WireSpec, ComponentSpec } from '@/types/graph';

/**
 * Add a new node to the graph.
 * Pure function - returns new graph spec.
 */
export function addNode(
  graph: GraphSpec,
  name: string,
  component: ComponentSpec,
): GraphSpec {
  if (name in graph.nodes) {
    throw new Error(`Node '${name}' already exists`);
  }
  return {
    ...graph,
    nodes: { ...graph.nodes, [name]: component },
  };
}

/**
 * Remove a node and all its wires.
 */
export function removeNode(graph: GraphSpec, name: string): GraphSpec {
  if (!(name in graph.nodes)) {
    throw new Error(`Node '${name}' does not exist`);
  }

  const newNodes = { ...graph.nodes };
  delete newNodes[name];

  const newWires = graph.wires.filter(
    w => w.source_node !== name && w.target_node !== name
  );

  const newInputBindings = { ...graph.input_bindings };
  const newOutputBindings = { ...graph.output_bindings };

  for (const [key, [nodeName]] of Object.entries(newInputBindings)) {
    if (nodeName === name) delete newInputBindings[key];
  }
  for (const [key, [nodeName]] of Object.entries(newOutputBindings)) {
    if (nodeName === name) delete newOutputBindings[key];
  }

  return {
    ...graph,
    nodes: newNodes,
    wires: newWires,
    input_bindings: newInputBindings,
    output_bindings: newOutputBindings,
  };
}

/**
 * Add a wire between two ports.
 */
export function addWire(graph: GraphSpec, wire: WireSpec): GraphSpec {
  // Validate source exists
  const sourceNode = graph.nodes[wire.source_node];
  if (!sourceNode) {
    throw new Error(`Source node '${wire.source_node}' does not exist`);
  }
  if (!sourceNode.output_ports.includes(wire.source_port)) {
    throw new Error(`Source port '${wire.source_port}' does not exist on '${wire.source_node}'`);
  }

  // Validate target exists
  const targetNode = graph.nodes[wire.target_node];
  if (!targetNode) {
    throw new Error(`Target node '${wire.target_node}' does not exist`);
  }
  if (!targetNode.input_ports.includes(wire.target_port)) {
    throw new Error(`Target port '${wire.target_port}' does not exist on '${wire.target_node}'`);
  }

  // Check for duplicate
  const exists = graph.wires.some(
    w => w.source_node === wire.source_node &&
         w.source_port === wire.source_port &&
         w.target_node === wire.target_node &&
         w.target_port === wire.target_port
  );
  if (exists) {
    throw new Error('Wire already exists');
  }

  return {
    ...graph,
    wires: [...graph.wires, wire],
  };
}

/**
 * Remove a wire.
 */
export function removeWire(graph: GraphSpec, wire: WireSpec): GraphSpec {
  return {
    ...graph,
    wires: graph.wires.filter(
      w => !(w.source_node === wire.source_node &&
             w.source_port === wire.source_port &&
             w.target_node === wire.target_node &&
             w.target_port === wire.target_port)
    ),
  };
}

/**
 * Insert a node between an existing wire.
 */
export function insertBetween(
  graph: GraphSpec,
  nodeName: string,
  component: ComponentSpec,
  wire: WireSpec,
  inputPort: string = 'input',
  outputPort: string = 'output',
): GraphSpec {
  let newGraph = removeWire(graph, wire);
  newGraph = addNode(newGraph, nodeName, component);
  newGraph = addWire(newGraph, {
    source_node: wire.source_node,
    source_port: wire.source_port,
    target_node: nodeName,
    target_port: inputPort,
  });
  newGraph = addWire(newGraph, {
    source_node: nodeName,
    source_port: outputPort,
    target_node: wire.target_node,
    target_port: wire.target_port,
  });
  return newGraph;
}
```

### 8.2 Graph Validation

```typescript
// features/graph/validation.ts

interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  cycles: string[][];  // List of cycle paths
}

interface ValidationError {
  type: 'missing_input' | 'invalid_wire' | 'duplicate_wire' | 'unbound_port';
  message: string;
  location?: { node?: string; port?: string; wire?: WireSpec };
}

interface ValidationWarning {
  type: 'unconnected_output' | 'potential_type_mismatch';
  message: string;
  location?: { node?: string; port?: string };
}

/**
 * Validate a graph for structural correctness.
 */
export function validateGraph(graph: GraphSpec): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  // Check all required inputs are connected
  for (const [nodeName, node] of Object.entries(graph.nodes)) {
    for (const inputPort of node.input_ports) {
      const hasWire = graph.wires.some(
        w => w.target_node === nodeName && w.target_port === inputPort
      );
      const hasBinding = Object.values(graph.input_bindings).some(
        ([n, p]) => n === nodeName && p === inputPort
      );

      if (!hasWire && !hasBinding) {
        errors.push({
          type: 'missing_input',
          message: `Input port '${nodeName}.${inputPort}' is not connected`,
          location: { node: nodeName, port: inputPort },
        });
      }
    }

    // Warn about unconnected outputs
    for (const outputPort of node.output_ports) {
      const hasWire = graph.wires.some(
        w => w.source_node === nodeName && w.source_port === outputPort
      );
      const hasBinding = Object.values(graph.output_bindings).some(
        ([n, p]) => n === nodeName && p === outputPort
      );

      if (!hasWire && !hasBinding) {
        warnings.push({
          type: 'unconnected_output',
          message: `Output port '${nodeName}.${outputPort}' is not connected`,
          location: { node: nodeName, port: outputPort },
        });
      }
    }
  }

  // Detect cycles
  const cycles = detectCycles(graph);

  return {
    valid: errors.length === 0,
    errors,
    warnings,
    cycles,
  };
}

/**
 * Detect cycles in the graph.
 * Returns list of node names forming each cycle.
 */
function detectCycles(graph: GraphSpec): string[][] {
  // Build adjacency list
  const adjacency: Record<string, Set<string>> = {};
  for (const nodeName of Object.keys(graph.nodes)) {
    adjacency[nodeName] = new Set();
  }
  for (const wire of graph.wires) {
    adjacency[wire.source_node].add(wire.target_node);
  }

  // DFS-based cycle detection
  const cycles: string[][] = [];
  const visited = new Set<string>();
  const recursionStack = new Set<string>();
  const path: string[] = [];

  function dfs(node: string): void {
    visited.add(node);
    recursionStack.add(node);
    path.push(node);

    for (const neighbor of adjacency[node]) {
      if (!visited.has(neighbor)) {
        dfs(neighbor);
      } else if (recursionStack.has(neighbor)) {
        // Found a cycle
        const cycleStart = path.indexOf(neighbor);
        cycles.push(path.slice(cycleStart));
      }
    }

    path.pop();
    recursionStack.delete(node);
  }

  for (const node of Object.keys(graph.nodes)) {
    if (!visited.has(node)) {
      dfs(node);
    }
  }

  return cycles;
}
```

### 8.3 Undo/Redo System

```typescript
// stores/historyStore.ts

import { create } from 'zustand';
import { GraphSpec } from '@/types/graph';

interface HistoryState {
  past: GraphSpec[];
  present: GraphSpec;
  future: GraphSpec[];

  push: (graph: GraphSpec) => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;
}

const MAX_HISTORY = 50;

export const useHistoryStore = create<HistoryState>((set, get) => ({
  past: [],
  present: createEmptyGraph(),
  future: [],

  push: (graph) => set(state => ({
    past: [...state.past.slice(-MAX_HISTORY + 1), state.present],
    present: graph,
    future: [],  // Clear redo stack on new action
  })),

  undo: () => set(state => {
    if (state.past.length === 0) return state;
    const previous = state.past[state.past.length - 1];
    return {
      past: state.past.slice(0, -1),
      present: previous,
      future: [state.present, ...state.future],
    };
  }),

  redo: () => set(state => {
    if (state.future.length === 0) return state;
    const next = state.future[0];
    return {
      past: [...state.past, state.present],
      present: next,
      future: state.future.slice(1),
    };
  }),

  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,
}));
```

---

## 9. Component System

### 9.1 Built-in Component Categories

| Category | Components |
|----------|------------|
| **Neural Networks** | SimpleStagedNetwork, MLP, GRU, LSTM |
| **Mechanics** | TwoLinkArm, PointMass, Spring, Damper |
| **Channels** | FeedbackChannel, DelayLine, NoiseInjector |
| **Interventions** | CurlField, ForceField, Clamp, Perturbation |
| **Tasks** | ReachingTask, TrackingTask, HoldTask |
| **Math** | Sum, Multiply, Gain, Saturation |
| **Signals** | Constant, Ramp, Sine, Pulse |

### 9.2 Component Library UI

```tsx
// components/panels/ComponentLibrary.tsx

export function ComponentLibrary() {
  const { components, isLoading } = useComponents();
  const [search, setSearch] = useState('');
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set(['Neural Networks']));

  const byCategory = useMemo(() => {
    const filtered = search
      ? components.filter(c =>
          c.name.toLowerCase().includes(search.toLowerCase()) ||
          c.description.toLowerCase().includes(search.toLowerCase())
        )
      : components;

    return groupBy(filtered, 'category');
  }, [components, search]);

  return (
    <div className="flex flex-col h-full">
      {/* Search */}
      <div className="p-3 border-b">
        <input
          type="text"
          placeholder="Search components..."
          value={search}
          onChange={e => setSearch(e.target.value)}
          className="w-full px-3 py-2 text-sm border rounded-md"
        />
      </div>

      {/* Category list */}
      <div className="flex-1 overflow-y-auto">
        {Object.entries(byCategory).map(([category, comps]) => (
          <CategorySection
            key={category}
            category={category}
            components={comps}
            expanded={expandedCategories.has(category)}
            onToggle={() => toggleCategory(category)}
          />
        ))}
      </div>
    </div>
  );
}

function ComponentCard({ component }: { component: ComponentDefinition }) {
  // Drag source for React DnD
  const [{ isDragging }, drag] = useDrag(() => ({
    type: 'COMPONENT',
    item: { component },
    collect: monitor => ({
      isDragging: monitor.isDragging(),
    }),
  }));

  return (
    <div
      ref={drag}
      className={`
        p-2 rounded-md cursor-grab border
        hover:bg-gray-50 hover:border-gray-300
        ${isDragging ? 'opacity-50' : ''}
      `}
    >
      <div className="flex items-center gap-2">
        <Icon name={component.icon} size={16} />
        <span className="text-sm font-medium">{component.name}</span>
      </div>
      <p className="text-xs text-gray-500 mt-1 line-clamp-2">
        {component.description}
      </p>
    </div>
  );
}
```

### 9.3 Custom Component Definition

User-defined components live in `~/.feedbax/components/`:

```python
# ~/.feedbax/components/my_custom_network.py

from feedbax.graph import Component
from feedbax.web.decorators import register_component
from equinox.nn import State
from jaxtyping import PRNGKeyArray, PyTree

@register_component(
    name="MyCustomNetwork",
    category="Custom",
    description="A custom neural network with special architecture",
    param_schema=[
        {"name": "layer_sizes", "type": "array", "default": [64, 64], "required": True},
        {"name": "activation", "type": "enum", "options": ["relu", "tanh", "gelu"], "default": "relu"},
    ],
)
class MyCustomNetwork(Component):
    input_ports = ("input",)
    output_ports = ("output", "hidden")

    layer_sizes: list[int]
    activation: str
    # ... internal weights, etc.

    def __call__(
        self,
        inputs: dict[str, PyTree],
        state: State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[dict[str, PyTree], State]:
        # Implementation...
        pass
```

### 9.4 Component Properties Panel

```tsx
// components/panels/PropertiesPanel.tsx

export function PropertiesPanel() {
  const selectedNode = useGraphStore(state => state.selectedNode);
  const updateNodeParams = useGraphStore(state => state.updateNodeParams);
  const { component } = useComponent(selectedNode?.type);

  if (!selectedNode || !component) {
    return (
      <div className="p-4 text-center text-gray-500">
        Select a node to view properties
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Header */}
      <div>
        <h3 className="font-medium">{selectedNode.name}</h3>
        <p className="text-sm text-gray-500">{component.name}</p>
      </div>

      {/* Parameters */}
      <div className="space-y-3">
        {component.param_schema.map(param => (
          <ParamInput
            key={param.name}
            schema={param}
            value={selectedNode.params[param.name]}
            onChange={value => updateNodeParams(selectedNode.id, param.name, value)}
          />
        ))}
      </div>

      {/* Ports info */}
      <div className="pt-4 border-t">
        <h4 className="text-sm font-medium mb-2">Ports</h4>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <span className="text-gray-500">Inputs:</span>
            <ul className="ml-2">
              {component.input_ports.map(p => <li key={p}>{p}</li>)}
            </ul>
          </div>
          <div>
            <span className="text-gray-500">Outputs:</span>
            <ul className="ml-2">
              {component.output_ports.map(p => <li key={p}>{p}</li>)}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

function ParamInput({ schema, value, onChange }: ParamInputProps) {
  switch (schema.type) {
    case 'int':
    case 'float':
      return (
        <NumberInput
          label={schema.name}
          value={value}
          onChange={onChange}
          min={schema.min}
          max={schema.max}
          step={schema.step ?? (schema.type === 'int' ? 1 : 0.01)}
        />
      );
    case 'bool':
      return (
        <Checkbox
          label={schema.name}
          checked={value}
          onChange={onChange}
        />
      );
    case 'enum':
      return (
        <Select
          label={schema.name}
          value={value}
          options={schema.options}
          onChange={onChange}
        />
      );
    // ... other types
  }
}
```

---

## 10. Training Configuration

### 10.1 Training Panel Structure

```
┌─────────────────────────────────────┐
│ Training                            │
├─────────────────────────────────────┤
│ ▼ Optimizer                         │
│   Type: [Adam ▼]                    │
│   Learning rate: [0.001    ]        │
│   β₁: [0.9  ]  β₂: [0.999]          │
│                                     │
│ ▼ Loss Function                     │
│   ┌───────────────────────────────┐ │
│   │ + Composite (1.0)             │ │
│   │   ├─ Position Error (1.0)     │ │
│   │   ├─ Velocity Penalty (0.1)   │ │
│   │   └─ Effort Cost (0.01)       │ │
│   └───────────────────────────────┘ │
│   [+ Add Term]                      │
│                                     │
│ ▼ Training Parameters               │
│   Batches: [1000     ]              │
│   Batch size: [64    ]              │
│   Checkpoint every: [100] batches   │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │  ▶ Start Training               │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

### 10.2 Loss Function Builder

The loss function is a tree structure (matching `TermTree` in Feedbax):

```tsx
// components/panels/LossBuilder.tsx

interface LossNode {
  id: string;
  type: string;
  weight: number;
  params: Record<string, any>;
  children?: LossNode[];
}

export function LossBuilder() {
  const [root, setRoot] = useState<LossNode>(defaultLoss);

  return (
    <div className="space-y-2">
      <LossTreeNode
        node={root}
        depth={0}
        onUpdate={setRoot}
        onRemove={() => {}}
        canRemove={false}
      />
      <button
        onClick={() => addChild(root)}
        className="text-sm text-blue-600 hover:text-blue-800"
      >
        + Add term
      </button>
    </div>
  );
}

function LossTreeNode({ node, depth, onUpdate, onRemove, canRemove }: LossTreeNodeProps) {
  const isComposite = node.children && node.children.length > 0;
  const indent = depth * 16;

  return (
    <div style={{ marginLeft: indent }}>
      <div className="flex items-center gap-2 p-2 rounded hover:bg-gray-50">
        {/* Expand/collapse for composites */}
        {isComposite && (
          <button onClick={() => toggleExpanded()}>
            {expanded ? '▼' : '▶'}
          </button>
        )}

        {/* Type selector */}
        <Select
          value={node.type}
          options={lossTypes}
          onChange={type => onUpdate({ ...node, type })}
          className="text-sm"
        />

        {/* Weight */}
        <NumberInput
          value={node.weight}
          onChange={weight => onUpdate({ ...node, weight })}
          min={0}
          step={0.1}
          className="w-16 text-sm"
        />

        {/* Remove button */}
        {canRemove && (
          <button onClick={onRemove} className="text-red-500">
            ×
          </button>
        )}
      </div>

      {/* Children */}
      {isComposite && expanded && (
        <div className="border-l ml-4 pl-2">
          {node.children.map((child, i) => (
            <LossTreeNode
              key={child.id}
              node={child}
              depth={depth + 1}
              onUpdate={updated => updateChild(i, updated)}
              onRemove={() => removeChild(i)}
              canRemove={true}
            />
          ))}
          <button onClick={() => addChild()} className="text-sm text-blue-600 ml-4">
            + Add child
          </button>
        </div>
      )}
    </div>
  );
}
```

### 10.3 Training Progress UI

```tsx
// components/panels/TrainingProgress.tsx

export function TrainingProgress() {
  const { status, progress, metrics, start, stop, pause } = useTraining();

  return (
    <div className="space-y-4">
      {/* Progress bar */}
      <div>
        <div className="flex justify-between text-sm mb-1">
          <span>Batch {progress.batch} / {progress.totalBatches}</span>
          <span>{((progress.batch / progress.totalBatches) * 100).toFixed(1)}%</span>
        </div>
        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all"
            style={{ width: `${(progress.batch / progress.totalBatches) * 100}%` }}
          />
        </div>
      </div>

      {/* Live metrics */}
      <div className="grid grid-cols-2 gap-4">
        <MetricCard label="Loss" value={metrics.loss.toFixed(4)} trend={metrics.lossTrend} />
        <MetricCard label="Learning Rate" value={metrics.lr.toExponential(2)} />
      </div>

      {/* Loss chart (mini) */}
      <div className="h-32">
        <LossChart data={metrics.lossHistory} />
      </div>

      {/* Controls */}
      <div className="flex gap-2">
        {status === 'idle' && (
          <Button onClick={start} variant="primary">
            ▶ Start Training
          </Button>
        )}
        {status === 'running' && (
          <>
            <Button onClick={pause} variant="secondary">
              ⏸ Pause
            </Button>
            <Button onClick={stop} variant="danger">
              ⏹ Stop
            </Button>
          </>
        )}
        {status === 'paused' && (
          <>
            <Button onClick={start} variant="primary">
              ▶ Resume
            </Button>
            <Button onClick={stop} variant="danger">
              ⏹ Stop
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
```

---

## 11. Execution & Visualization

### 11.1 Simulation Preview

Users can run a quick simulation to see model behavior:

```tsx
// components/panels/SimulationPreview.tsx

export function SimulationPreview() {
  const { runSimulation, isRunning, result } = useSimulation();
  const [nSteps, setNSteps] = useState(100);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <NumberInput
          label="Steps"
          value={nSteps}
          onChange={setNSteps}
          min={1}
          max={1000}
        />
        <Button onClick={() => runSimulation(nSteps)} disabled={isRunning}>
          {isRunning ? 'Running...' : 'Run'}
        </Button>
      </div>

      {result && (
        <div className="space-y-4">
          {/* Trajectory plot */}
          <div className="h-48">
            <TrajectoryPlot data={result.trajectory} />
          </div>

          {/* State inspector */}
          <StateInspector states={result.stateHistory} />
        </div>
      )}
    </div>
  );
}
```

### 11.2 Data Flow Visualization

During execution, show data flowing through wires:

```tsx
// Animated edges during simulation

function AnimatedEdge({ data, ...props }: EdgeProps) {
  const isSimulating = useExecutionStore(state => state.isSimulating);
  const portValue = useExecutionStore(
    state => state.portValues[`${data.sourceNode}.${data.sourcePort}`]
  );

  return (
    <BaseEdge {...props}>
      {isSimulating && portValue !== undefined && (
        <>
          {/* Animated dots along edge */}
          <circle r="4" fill="#3b82f6">
            <animateMotion dur="1s" repeatCount="indefinite">
              <mpath href={`#${props.id}`} />
            </animateMotion>
          </circle>

          {/* Value tooltip at midpoint */}
          <foreignObject x={midX - 30} y={midY - 10} width={60} height={20}>
            <div className="text-xs bg-black/80 text-white px-1 rounded text-center">
              {formatValue(portValue)}
            </div>
          </foreignObject>
        </>
      )}
    </BaseEdge>
  );
}
```

### 11.3 State Inspector

Inspect component state during/after execution:

```tsx
// components/panels/StateInspector.tsx

export function StateInspector() {
  const selectedNode = useGraphStore(state => state.selectedNode);
  const nodeState = useExecutionStore(
    state => selectedNode ? state.nodeStates[selectedNode.id] : null
  );

  if (!nodeState) {
    return <div className="text-gray-500 text-sm">No state to inspect</div>;
  }

  return (
    <div className="font-mono text-xs">
      <TreeView data={nodeState} />
    </div>
  );
}

function TreeView({ data, depth = 0 }: { data: any; depth?: number }) {
  if (data === null || data === undefined) {
    return <span className="text-gray-400">null</span>;
  }

  if (typeof data === 'number') {
    return <span className="text-blue-600">{data.toPrecision(4)}</span>;
  }

  if (Array.isArray(data)) {
    // Check if it's a JAX array (show shape + sample values)
    if (data.length > 10) {
      return (
        <span className="text-green-600">
          Array[{data.length}]: [{data.slice(0, 3).map(v => v.toPrecision(2)).join(', ')}, ...]
        </span>
      );
    }
    return (
      <span>[{data.map((v, i) => (
        <span key={i}>
          {i > 0 && ', '}
          <TreeView data={v} depth={depth + 1} />
        </span>
      ))}]</span>
    );
  }

  if (typeof data === 'object') {
    return (
      <div style={{ marginLeft: depth * 12 }}>
        {Object.entries(data).map(([key, value]) => (
          <div key={key}>
            <span className="text-purple-600">{key}</span>: <TreeView data={value} depth={depth + 1} />
          </div>
        ))}
      </div>
    );
  }

  return <span>{String(data)}</span>;
}
```

---

## 12. State Management

### 12.1 Store Architecture

```typescript
// stores/graphStore.ts

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { Node, Edge, OnNodesChange, OnEdgesChange, OnConnect, applyNodeChanges, applyEdgeChanges, addEdge } from '@xyflow/react';
import { GraphSpec, ComponentSpec, WireSpec } from '@/types/graph';
import * as ops from '@/features/graph/operations';

interface GraphState {
  // Core data
  graphSpec: GraphSpec;

  // React Flow state (derived but managed for performance)
  nodes: Node[];
  edges: Edge[];

  // Selection
  selectedNodeId: string | null;
  selectedEdgeId: string | null;

  // Actions
  setGraph: (spec: GraphSpec) => void;
  addNode: (name: string, component: ComponentSpec, position: { x: number; y: number }) => void;
  removeNode: (name: string) => void;
  addWire: (wire: WireSpec) => void;
  removeWire: (wire: WireSpec) => void;
  updateNodeParams: (name: string, paramName: string, value: any) => void;

  // React Flow handlers
  onNodesChange: OnNodesChange;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;

  // Selection
  selectNode: (id: string | null) => void;
  selectEdge: (id: string | null) => void;
}

export const useGraphStore = create<GraphState>()(
  devtools(
    persist(
      (set, get) => ({
        graphSpec: createEmptyGraph(),
        nodes: [],
        edges: [],
        selectedNodeId: null,
        selectedEdgeId: null,

        setGraph: (spec) => {
          const { nodes, edges } = specToReactFlow(spec);
          set({ graphSpec: spec, nodes, edges });
        },

        addNode: (name, component, position) => {
          const newSpec = ops.addNode(get().graphSpec, name, component);
          const node = componentToNode(name, component, position);
          set({
            graphSpec: newSpec,
            nodes: [...get().nodes, node],
          });
          // Push to history
          useHistoryStore.getState().push(newSpec);
        },

        removeNode: (name) => {
          const newSpec = ops.removeNode(get().graphSpec, name);
          set({
            graphSpec: newSpec,
            nodes: get().nodes.filter(n => n.id !== name),
            edges: get().edges.filter(e => e.source !== name && e.target !== name),
          });
          useHistoryStore.getState().push(newSpec);
        },

        addWire: (wire) => {
          const newSpec = ops.addWire(get().graphSpec, wire);
          const edge = wireToEdge(wire);
          set({
            graphSpec: newSpec,
            edges: [...get().edges, edge],
          });
          useHistoryStore.getState().push(newSpec);
        },

        // ... other actions

        onNodesChange: (changes) => {
          set({ nodes: applyNodeChanges(changes, get().nodes) });
        },

        onEdgesChange: (changes) => {
          set({ edges: applyEdgeChanges(changes, get().edges) });
        },

        onConnect: (connection) => {
          const wire: WireSpec = {
            source_node: connection.source!,
            source_port: connection.sourceHandle!,
            target_node: connection.target!,
            target_port: connection.targetHandle!,
          };
          get().addWire(wire);
        },
      }),
      { name: 'feedbax-graph' }
    )
  )
);
```

### 12.2 Derived State

```typescript
// hooks/useGraph.ts

import { useMemo } from 'react';
import { useGraphStore } from '@/stores/graphStore';
import { validateGraph, detectCycles } from '@/features/graph/validation';

export function useGraphValidation() {
  const graphSpec = useGraphStore(state => state.graphSpec);

  return useMemo(() => validateGraph(graphSpec), [graphSpec]);
}

export function useSelectedNode() {
  const selectedNodeId = useGraphStore(state => state.selectedNodeId);
  const graphSpec = useGraphStore(state => state.graphSpec);

  return useMemo(() => {
    if (!selectedNodeId) return null;
    const component = graphSpec.nodes[selectedNodeId];
    if (!component) return null;
    return { id: selectedNodeId, name: selectedNodeId, ...component };
  }, [selectedNodeId, graphSpec.nodes]);
}

export function useCycleWires() {
  const graphSpec = useGraphStore(state => state.graphSpec);

  return useMemo(() => {
    const cycles = detectCycles(graphSpec);
    const cycleWires = new Set<string>();

    for (const cycle of cycles) {
      for (let i = 0; i < cycle.length; i++) {
        const from = cycle[i];
        const to = cycle[(i + 1) % cycle.length];
        // Find wires between these nodes
        for (const wire of graphSpec.wires) {
          if (wire.source_node === from && wire.target_node === to) {
            cycleWires.add(`${wire.source_node}.${wire.source_port}->${wire.target_node}.${wire.target_port}`);
          }
        }
      }
    }

    return cycleWires;
  }, [graphSpec]);
}
```

---

## 13. API Specification

### 13.1 REST Endpoints

```yaml
# OpenAPI-style specification

/api/graphs:
  GET:
    summary: List all saved graphs
    response: { graphs: GraphMetadata[] }

  POST:
    summary: Create a new graph
    body: GraphSpec
    response: { id: string, metadata: GraphMetadata }

/api/graphs/{id}:
  GET:
    summary: Get a graph by ID
    response: { graph: GraphSpec, ui_state: GraphUIState }

  PUT:
    summary: Update a graph
    body: { graph?: GraphSpec, ui_state?: GraphUIState }
    response: { success: boolean }

  DELETE:
    summary: Delete a graph
    response: { success: boolean }

/api/graphs/{id}/validate:
  POST:
    summary: Validate a graph
    body: GraphSpec
    response: ValidationResult

/api/graphs/{id}/export:
  POST:
    summary: Export graph as Python code or JSON
    body: { format: "json" | "python" }
    response: { content: string, filename: string }

/api/components:
  GET:
    summary: List all available components
    response: { components: ComponentDefinition[] }

/api/components/{name}:
  GET:
    summary: Get component details
    response: ComponentDefinition

/api/components/refresh:
  POST:
    summary: Reload user components from disk
    response: { added: string[], removed: string[] }

/api/training:
  POST:
    summary: Start a training job
    body: { graph_id: string, training_spec: TrainingSpec, task_spec: TaskSpec }
    response: { job_id: string }

/api/training/{job_id}:
  GET:
    summary: Get training job status
    response: TrainingStatus

  DELETE:
    summary: Stop a training job
    response: { success: boolean }

/api/training/{job_id}/checkpoint:
  GET:
    summary: Get latest checkpoint
    response: { checkpoint_path: string, batch: number }

/api/execution/simulate:
  POST:
    summary: Run a simulation
    body: { graph: GraphSpec, n_steps: number, inputs?: Record<string, any> }
    response: { outputs: Record<string, any>, state_history?: any }
```

### 13.2 WebSocket Messages

```typescript
// Client -> Server

interface StartTrainingMessage {
  type: 'start_training';
  job_id: string;
}

interface StopTrainingMessage {
  type: 'stop_training';
  job_id: string;
}

interface SimulateStepMessage {
  type: 'simulate_step';
  graph_id: string;
}

// Server -> Client

interface TrainingProgressMessage {
  type: 'training_progress';
  job_id: string;
  batch: number;
  total_batches: number;
  loss: number;
  metrics: Record<string, number>;
}

interface TrainingCompleteMessage {
  type: 'training_complete';
  job_id: string;
  final_loss: number;
  checkpoint_path: string;
}

interface TrainingErrorMessage {
  type: 'training_error';
  job_id: string;
  error: string;
}

interface SimulationStateMessage {
  type: 'simulation_state';
  step: number;
  port_values: Record<string, any>;
  node_states: Record<string, any>;
}
```

---

## 14. File Structure

### 14.1 Project Layout

```
feedbax/
├── feedbax/                    # Python library (existing)
│   ├── graph.py
│   ├── ...
│   └── web/                    # NEW: Web backend
│       ├── __init__.py
│       ├── app.py
│       ├── api/
│       ├── ws/
│       ├── services/
│       ├── models/
│       └── serialization.py
│
├── web/                        # NEW: Frontend application
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.js
│   ├── index.html
│   ├── public/
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── components/
│       ├── features/
│       ├── stores/
│       ├── api/
│       ├── hooks/
│       ├── types/
│       └── utils/
│
├── docs/
│   └── WEB_UI_SPEC.md          # This document
│
└── scripts/
    ├── dev.sh                  # Start frontend + backend
    └── build.sh                # Production build
```

### 14.2 Configuration Files

```javascript
// web/vite.config.ts
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
```

```javascript
// web/tailwind.config.js
module.exports = {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        // Custom colors if needed
      },
    },
  },
  plugins: [],
};
```

---

## 15. Implementation Phases

### Phase 1: Foundation (2-3 weeks)

**Goal**: Basic canvas with node display and wiring

- [ ] Project scaffolding (Vite + React + TypeScript + Tailwind)
- [ ] React Flow integration with custom node components
- [ ] Basic graph store (Zustand)
- [ ] Component library panel (hardcoded components)
- [ ] Drag-and-drop node creation
- [ ] Wire creation between ports
- [ ] Basic FastAPI backend scaffold
- [ ] Graph serialization (JSON ↔ Python)

**Deliverable**: Can create a simple graph visually and export to JSON

### Phase 2: Component System (1-2 weeks)

**Goal**: Full component registry and properties editing

- [ ] Component registry (backend)
- [ ] Component API endpoints
- [ ] Properties panel with parameter editing
- [ ] Input validation and type handling
- [ ] User component discovery
- [ ] Graph validation (frontend + backend)
- [ ] Cycle detection and visualization

**Deliverable**: Can configure all built-in components via UI

### Phase 3: Training Integration (2-3 weeks)

**Goal**: Configure and run training from UI

- [ ] Training configuration panel
- [ ] Optimizer configuration
- [ ] Loss function builder (tree UI)
- [ ] WebSocket for training progress
- [ ] Training progress visualization
- [ ] Start/stop/pause controls
- [ ] Checkpoint management

**Deliverable**: Can train a model from the UI

### Phase 4: Execution & Debugging (1-2 weeks)

**Goal**: Run simulations and inspect state

- [ ] Simulation preview panel
- [ ] State inspector
- [ ] Data flow visualization (animated edges)
- [ ] Port value display
- [ ] Step-through execution mode

**Deliverable**: Can debug model behavior visually

### Phase 5: Polish & Integration (1-2 weeks)

**Goal**: Production-ready experience

- [ ] Undo/redo system
- [ ] Keyboard shortcuts
- [ ] File management (save, load, export)
- [ ] Error handling and user feedback
- [ ] Dark mode
- [ ] Performance optimization
- [ ] Documentation

**Deliverable**: Polished, usable application

### Phase 6: Future Enhancements (Ongoing)

- [ ] Analysis dashboard integration
- [ ] In-browser code editor for custom components
- [ ] Model versioning and comparison
- [ ] Collaboration features
- [ ] Cloud storage integration

---

## 16. Future Considerations

### 16.1 Analysis Integration

The existing Dash-based dashboard will eventually be absorbed into this application as additional views:

- **Figures view**: Display and filter analysis figures (existing functionality)
- **Experiment browser**: Navigate saved experiments and models
- **Comparison view**: Side-by-side model comparison

These would be separate routes/tabs in the same application, sharing the component system and styling.

### 16.2 Meta-Canvas View

A higher-level view showing the relationship between:
- TaskTrainer
- Model (Graph)
- Task
- Analysis pipelines

This would be a separate canvas mode for understanding the overall workflow, not the model internals.

### 16.3 Custom Component Editor

A future enhancement could provide in-browser editing:
- Code editor with syntax highlighting (Monaco)
- Template scaffolding for new components
- Validation and error feedback
- Hot-reload during development

### 16.4 Real-time Collaboration

If multiple users need to work on the same model:
- Operational transform or CRDT for conflict resolution
- Presence indicators (who is viewing what)
- Comments and annotations
- Change history

### 16.5 Cloud Deployment

For team/production use:
- Docker containerization
- Cloud storage for models and checkpoints
- GPU instances for training
- Authentication and authorization

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Component** | A node in the computational graph; base class for all graph elements |
| **Wire** | A connection from an output port to an input port |
| **Graph** | A collection of components and wires; itself a component (composable) |
| **Port** | A named input or output on a component |
| **StateIndex** | Equinox pattern for managing persistent state in JAX |
| **TermTree** | Hierarchical loss function structure |
| **TaskTrainer** | Orchestrator for training loops |
| **Intervention** | Modification to model behavior (e.g., force field, perturbation) |

---

## Appendix B: References

- [React Flow Documentation](https://reactflow.dev)
- [Collimator](https://www.collimator.ai/) - Similar product for inspiration
- [Zustand](https://github.com/pmndrs/zustand) - State management
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [Tailwind CSS](https://tailwindcss.com/) - Styling
- [Radix UI](https://www.radix-ui.com/) - Accessible components

---

*Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>*
