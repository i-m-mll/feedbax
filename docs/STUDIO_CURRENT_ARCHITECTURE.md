# Feedbax Studio current architecture

## Mounted application

The browser and desktop Studio share the React application rooted at [`web/src/main.tsx`](../web/src/main.tsx) and [`App.tsx`](../web/src/App.tsx). The browser root creates a managed project through `POST /api/projects` and puts its ID in the URL. `/?project=<id>` opens an existing project. `/?activity=1` opens standalone Activity. The desktop File picker and Finder entry points open project folders or declarations through the native control socket and take the webview to the same managed project route. The desktop shell and control transport are described in [`desktop/README.md`](../desktop/README.md).

[`SavedTrainWorkspace.tsx`](../web/src/features/projectFolder/SavedTrainWorkspace.tsx) opens one project editor and supplies [`ProjectWindow.tsx`](../web/src/features/projectWindow/ProjectWindow.tsx) with the project’s Plan, selection, views, and commands. The map and its drawer, the two Details slots, and the inspector share a variant-qualified selection with Back and Forward history. The available tabs follow the selected step and its data. Train exposes Runs, Model, Task, Objective, Training, Scene, and Results as applicable. Evaluate uses Runs, Results, and Scene for saved evaluation and replay where available. Analyze exposes Analysis and Figures, and Report exposes Document. Activity follows the shared Jobs stream and can open a run in its project.

The managed Model canvas is in [`web/src/features/projectFolder/model/`](../web/src/features/projectFolder/model/). [`managedProjectDetails.tsx`](../web/src/features/projectFolder/managedProjectDetails.tsx) connects the saved Model, Task, and Scene projections to mounted views. Scene uses the Scenario projection and replay types and domain-specific model data. [`studioCommands.ts`](../web/src/features/projectWindow/studioCommands.ts) defines the shared command bindings. The current selection, focus, and command availability govern the palette, keyboard actions, and native project menus.

## Authored files, Plan, and Jobs

The [text project codec](../feedbax/contracts/project_io.py) defines the folder format. A single `*.feedbax` declaration names the project and its Store. Authored JSON files hold documents and steps. `presentation.json` holds shared layout independently of scientific identity. [`openProjectEditor.ts`](../web/src/features/projectFolder/openProjectEditor.ts) and [`projectEditController.ts`](../web/src/features/projectFolder/projectEditController.ts) manage optimistic edits, revision-checked saves, Undo and Redo, and reconciliation against outside edits. Pane modes and dimensions are window preferences. The project-folder service verifies and applies writes in the [core folder implementation](../feedbax/project_folder.py).

[`/api/projects`](../feedbax/web/api/projects.py) serves open, authoring, Model operations, projection, history replay, and presentation. [`ProjectPlanService`](../feedbax/web/services/project_plan.py) compiles the acknowledged folder revision and joins its run keys to Jobs attempts. The managed window requests that projection through `GET /api/projects/{id}/train-runs?revision=…&window_map=true`. The Plan supplies the map, runs, coordinates, lineage, and Problems. The Jobs routes in [`feedbax/web/api/jobs.py`](../feedbax/web/api/jobs.py) handle admission, Stop, Resume, snapshots, and replayable server-sent events. The [shared browser Jobs client](../web/src/jobs/studioJobWorker.ts) follows the origin’s event stream and resynchronizes its projection after disconnects. The [local Host](../feedbax/host/daemon.py) owns execution and attempts beyond the browser or API process. Saved Evaluate uses the managed Plan and Jobs path.

TypeScript API contracts are generated from Python models by [`scripts/generate_studio_contracts.py`](../scripts/generate_studio_contracts.py). [`studioContracts.ts`](../web/src/generated/studioContracts.ts) is the generated output. The [Studio vocabulary table](../web/src/components/activity/vocabulary.json) supplies names, statuses, and reason templates for converted surfaces.

## Supported backend boundaries

[`feedbax/web/app.py`](../feedbax/web/app.py) is the public FastAPI route inventory. Components, domains, and provider registries supply authoring metadata. The Scenario task-trial sampler, domain registry, and MuJoCo asset intake retain their explicit endpoints. The managed window uses Plan and Jobs for training and activity. The retired Studio training and simulation transports have no routes in this app.

Analysis, figure, trajectory, statistics, and inspection routes are listed in the FastAPI route inventory. The managed Figures view comes from the selected authored analysis. The [downstream stability inventory](design/downstream_interface_stability.md) links the current managed interfaces and the historical ratification. See the [typed domain design](design/typed_subgraph_domains.md) for domain contracts.

## Verification

Follow the [source setup](../README.md) before running `scripts/dev.sh`. Its default launch creates a temporary home. `--home <dir>` keeps an explicit project and run home. `--preview` builds and serves the production browser bundle. Both modes proxy to the isolated loopback API. [`.claude/launch.json`](../.claude/launch.json) names both launch modes.

From `web/`, `bun scripts/check-reachability.ts` checks the production import graph rooted at `main.tsx` and checks for exported API functions without references in that graph. [CI](../.github/workflows/ci.yml) invokes it explicitly. `bun run test:studio <spec-or-pattern>` starts a real isolated backend through the [Studio Playwright configuration](../web/playwright.studio.config.ts). Set `FEEDBAX_STUDIO_E2E_PREVIEW=1` to use the production browser build. `bun run test:e2e` runs the separate mocked smoke configuration.

The older [web UI specification](WEB_UI_SPEC.md) and [design response](WEB_UI_RESPONSE_2026-01-27.md) record the architecture before the managed project window.
