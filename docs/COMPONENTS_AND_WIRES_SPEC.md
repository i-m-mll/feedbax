# Components & Wire Visualization Spec

This spec covers two related improvements: restoring lost components and implementing
dual wire representation for better state visualization.

---

## Part 1: Restore Lost Components

During the eager architecture merge, several components were removed from the registry.
These should be restored.

### Math Operations

| Component | Description | Ports |
|-----------|-------------|-------|
| **Gain** | Multiply input by constant | in: `input` / out: `output` |
| **Sum** | Add/subtract multiple inputs | in: `a`, `b` / out: `output` |
| **Multiply** | Element-wise product | in: `a`, `b` / out: `output` |

### Signal Sources

| Component | Description | Ports |
|-----------|-------------|-------|
| **Constant** | Constant value output | out: `output` |
| **Ramp** | Linear ramp over time | out: `output` |
| **Sine** | Sinusoidal signal | out: `output` |
| **Pulse** | Pulse/square wave | out: `output` |

### Neural Network Modules

| Component | Description | Ports |
|-----------|-------------|-------|
| **MLP** | Multi-layer perceptron (standalone) | in: `input` / out: `output` |
| **GRU** | GRU cell/network (standalone) | in: `input`, `hidden` / out: `output`, `hidden` |
| **LSTM** | LSTM cell/network (standalone) | in: `input`, `hidden`, `cell` / out: `output`, `hidden`, `cell` |

### Mechanical Primitives

| Component | Description | Ports |
|-----------|-------------|-------|
| **Spring** | Linear spring (F = k × displacement) | in: `displacement` / out: `force` |
| **Damper** | Viscous damper (F = b × velocity) | in: `velocity` / out: `force` |

### Signal Processing

| Component | Description | Ports |
|-----------|-------------|-------|
| **Saturation** | Clamp to min/max range | in: `input` / out: `output` |
| **DelayLine** | Discrete delay buffer | in: `input` / out: `output` |
| **Noise** | Random noise source | out: `output` |

### Implementation

Add these to `feedbax/web/services/component_registry.py` following the existing pattern.
Use appropriate categories: `Math`, `Sources`, `Neural Networks`, `Mechanics`, `Signal Processing`.

---

## Part 2: Dual Wire Representation

### Concept

Currently, wires show individual port connections. But since we pass the entire state
between components, there are conceptually two levels of data flow:

1. **State flow** - The entire state PyTree passing from component to component
2. **Port connections** - Which specific keys each component reads/writes

### Visual Design

**State flow wires (thick):**
- Thicker, darker line connecting nodes
- Shows the "main" execution path
- Connects from right side of one node to left side of next
- Visible even when nodes are collapsed
- Single port dot per side (larger than port dots)

**Port wires (thin):**
- Current thin lines connecting specific ports
- Only visible when nodes are expanded
- Show which state keys are being routed

### Example

```
COLLAPSED VIEW (state flow only):
┌─────────┐           ┌─────────┐           ┌─────────┐
│  Task   │━━━━━━━━━━▶│ Network │━━━━━━━━━━▶│Mechanics│
└─────────┘           └─────────┘           └─────────┘
     ◉                     ◉                     ◉
   (thick state-flow wire between large dots)

EXPANDED VIEW (state flow + port wires):
┌─────────────┐                 ┌─────────────┐
│ Task        │                 │ Network     │
├─────────────┤                 ├─────────────┤
│      inputs─●────────────────▶●─input       │
│     targets─●                 │      output─●────▶ ...
│       inits─●                 │      hidden─●
│   intervene─●                 │             │
│           ◉━━━━━━━━━━━━━━━━━━━◉             │
└─────────────┘                 └─────────────┘
   (large dot)   (thick line)    (large dot)
```

### Implementation Notes

1. **State flow port**: Add a special "state" port at header level (right side)
   - Larger dot (e.g., 10px vs 6px for regular ports)
   - Positioned at vertical center of header or collapsed body
   - Not user-connectable (automatic based on execution order)

2. **State flow edges**: New edge type `state-flow`
   - Thicker stroke (e.g., 3px vs 1.5px)
   - Darker color (e.g., `#475569` vs `#94a3b8`)
   - Always rendered (even when nodes collapsed)
   - Derived from topological sort of graph

3. **Collapsed behavior**:
   - Hide all port wires
   - Show only state-flow wires
   - Single large dot on each side

4. **React Flow integration**:
   - May need custom edge component for state-flow
   - Or render state-flow as separate layer

### Files to Modify

- `web/src/components/canvas/CustomNode.tsx` - Add state port rendering
- `web/src/components/canvas/RoutedEdge.tsx` - Add state-flow edge variant
- `web/src/stores/graphStore.ts` - Compute state-flow edges from execution order

---

## Summary

| Feature | Priority | Complexity |
|---------|----------|------------|
| Restore math ops (Gain, Sum, Multiply) | High | Low |
| Restore signal sources | Medium | Low |
| Restore NN modules (MLP, GRU, LSTM) | High | Low |
| Restore mechanical primitives | Medium | Low |
| Restore signal processing | Medium | Low |
| State-flow wire visualization | Medium | Medium |

---

*Co-authored by Claude Opus 4.5*
