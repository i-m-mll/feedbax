# Workspace View 3D Strategy Hedges

Status: ratified for [issue:b1ecca4] under umbrella [issue:f3159c7].

This note records the 3D strategy hedges for the Workspace view. It is
analysis-only: it does not define new runtime code, migrations, renderer
behavior, or saved graph compatibility.

## Ratified Decisions

1. Anchor frames are dimensioned data.

   Each anchor frame must declare `dim`. The current Workspace representation is
   2D, so `dim = 2` is the expected value today. Future 3D work should raise the
   dimension at the data-contract boundary instead of creating a renderer-local
   convention.

2. Replay tracks stay renderer-neutral.

   Replay products expose Cartesian tracks with shape `(T, dim)`. SVG, Pixi,
   three.js, rerun export, and any later renderer all consume the same replay
   product. A renderer may project, style, or hide dimensions, but it must not
   reinterpret the product shape as renderer-specific data.

3. Existing 2D archetypes may remain honestly 2D-specific.

   The 3D path does not require generalizing every current 2D archetype. Add 3D
   archetypes when the underlying representation is different. Expected future
   examples are:

   - `kinematic_tree` for joint/link structures with explicit frame topology.
   - `mesh_ref` for externally owned mesh assets addressed by stable refs.
   - `mjcf_scene` for passthrough MuJoCo visual metadata when a component already
     owns or emits MJCF.

4. The renderer swap point is scene construction versus frame application.

   Scene construction resolves entities, archetypes, visual metadata, selectors,
   and frame sources into a renderer-independent scene product. Frame application
   applies `(T, dim)` or live frame values onto that scene. A future 3D renderer
   should replace the renderer-specific construction and application surfaces,
   not the upstream representation contract.

5. MuJoCo-WASM is reserved for interactive browser simulation.

   MuJoCo-WASM should not be a dependency for Workspace replay or authoring.
   Replay and authoring only need already compiled scene metadata and Cartesian
   frame products. If Feedbax later needs interactive in-browser MuJoCo
   simulation, first-party MuJoCo JavaScript/WASM is now a plausible candidate,
   but it should enter as an explicit simulation feature with its own acceptance
   tests and browser constraints.

   Current source check, 2026-07-07:

   - Google DeepMind's MuJoCo repository documents first-party
     JavaScript/TypeScript WASM bindings and the `@mujoco/mujoco` package:
     <https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md>.
   - The same README marks the bindings as WIP, documents explicit memory
     management, distinguishes copy versus live-view data access, and notes
     browser constraints for the multi-threaded build.
   - The official package is published on npm as `@mujoco/mujoco`; `npm view
     @mujoco/mujoco` reported latest version `3.10.0` with description "MuJoCo
     WASM bindings" on 2026-07-07:
     <https://www.npmjs.com/package/@mujoco/mujoco>.
   - The repository's example application uses Three.js for browser rendering,
     which supports treating three.js as the likely first 3D visualization step
     even if MuJoCo-WASM later powers interactive simulation.

6. Interaction affordances are not future-proofed here.

   2D pan, zoom, drag handles, hit targets, labels, and direct-manipulation
   affordances may stay optimized for the current Workspace. The hedge is the
   data contract: dimensioned anchors, renderer-neutral replay tracks, and
   additive 3D archetypes.

## Implications for Later Children

- [issue:fd5143f] owns representation contract and schema code. It should treat
  `dim` and `(T, dim)` as contract decisions, not renderer preferences.
- [issue:525249b] owns muscle geometry defaults and accessors. This note does
  not change those APIs; it only reserves a future path for 3D geometry metadata.
- [issue:e2cc8ec] and [issue:9534977] may keep the first visible implementation
  2D/SVG-first while preserving the data-contract hedge above.

## Out of Scope

- No runtime code changes.
- No migration or schema implementation.
- No commitment to MuJoCo-WASM for replay, authoring, or first-pass 3D
  visualization.
- No claim that missing `DESIGN.md` content has been reconstructed.
