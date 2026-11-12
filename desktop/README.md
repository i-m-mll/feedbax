# Feedbax Studio for macOS

Feedbax Studio uses Tauri, macOS 14 or newer on Apple Silicon, and bundled CPython 3.12.9 with MJX on JAX CPU. Each Studio bundle contains Feedbax from the same source commit and version as the release. The app reads its resources from the signed bundle and requires no developer checkout, virtual environment, or dependency installer at runtime.

The native build uses the signed shared window shell pinned in `vendor/window-shell`. Initialize that submodule before building. For a local-only source, set the submodule URL for the checkout to the registered shell repository and initialize with Git's file transport allowed for that command. The native receipt records the pin.

## Build a private candidate

From a clean commit on an Apple Silicon build Mac with Xcode command-line tools, Rust, Bun and uv, install the two locked Bun environments and run the native and bundle phases.

```sh
(cd desktop && bun install --frozen-lockfile)
(cd web && bun install --frozen-lockfile)
uv run --no-project --with packaging==26.3 python desktop/package.py native
uv run --no-project --with packaging==26.3 python desktop/package.py bundle \
  --output .desktop-build/candidate --build-number 1 \
  --numerical-manifest "$STUDIO_NUMERICAL_MANIFEST" \
  --signing-identity "$STUDIO_SIGNING_IDENTITY"
```

Use an exact Apple Development certificate name or fingerprint for a private candidate. Passing `-` requests ad-hoc signing explicitly. Both are private build classes. Public distribution requires Developer ID, notarization and quarantine first-launch qualification. Allow a separate bounded window for the first Cargo compile.

The native phase uses Cargo's normal target reuse. Bundle assembly checks the recorded native inputs and launcher digest, then runs Tauri's bundle phase without recompilation. The release directory contains the signed ZIP and a receipt binding it to the source commit, native payload, wheel closure and signing identity. Use a new output path and increment the build number for a replacement release.

For a private fixture, add `--fixture-name <unique-name> --fixture-root <absolute-path>` to the bundle phase. Both releases in a replacement exercise must use the same fixture name and state root. Keep the absolute root short enough for the macOS Unix socket address limit.

Every bundle uses the locked `mjx` extra. Set `STUDIO_NUMERICAL_MANIFEST` to the qualified `feedbax.desktop.wheels.v1` manifest for `aarch64-apple-darwin`, with its wheel files staged alongside it. The manifest declares `runtime` and attests the exact filenames and SHA-256 values for `jax`, `jaxlib`, `mujoco-core` and `mujoco-mjx` under `wheels`. Bundle assembly requires these inputs to match the selected lock closure before it assembles the app. The sealed dependency inventory reports the MJX JAX provider on CPU.

## Install and replace

```sh
ditto -x -k .desktop-build/candidate/Feedbax-Studio-macos-arm64.zip "$STUDIO_UNPACK_DIR"
"$STUDIO_UNPACK_DIR/Feedbax Studio.app/Contents/MacOS/feedbax-studio" install "$STUDIO_APP_PATH" \
  --archive .desktop-build/candidate/Feedbax-Studio-macos-arm64.zip \
  --receipt .desktop-build/candidate/receipt.json --jobs defer
```

The install command verifies the supplied archive and signed app, registers its background item and waits for the installed service's exact release identity. Open the installed app through Finder after installation. For a replacement, quit Studio first. With `--jobs defer`, active jobs prevent replacement. Select `--jobs checkpoint` to request a cooperative stop or `--jobs force` to terminate active work explicitly. Preserve the release archive and receipt together.

Run the installer commands on the executable in the unpacked app. The executable uses its bundled interpreter and installer. Use `recover <app-path> --control <control-directory>` to continue or add `--rollback` to restore the previous bundle. The production control directory is `~/.feedbax/config/desktop-service`. A fixture uses `<fixture-root>/state/config/desktop-service`. If macOS requires background item approval, open Login Items in System Settings, enable the item and run recovery. The `uninstall <app-path>` operation stops and unregisters Studio and removes the app. The same job dispositions apply.

## Service and data

Studio's Quit command closes the GUI. Its per-user background service continues training and analysis. Reopen Studio and use Background jobs to inspect them. Cancel requests a cooperative training stop or interrupts an analysis job. A settled native training checkpoint can start a linked continuation. After a service crash, native training adopts verified surviving local workers, while interrupted HTTP worker jobs and analyses retain their last durable disposition and results.

The application menu has idle, checkpoint and force service stops, Reconnect, Login Items and Service Logs. Stop at a checkpoint can wait for the next supported training checkpoint. The owner-only control socket is separate from the loopback Studio origin, and the app exchanges a short-lived ticket for the browser session.

Installed files live inside the app bundle. Durable application state lives under `~/.feedbax`, with configuration in `config`, preferences in `settings` and the default workspace under `workspace`. Project-selected output locations retain models, checkpoints and results. `~/.cache/feedbax` contains disposable compilation files, temporary files and bounded service logs. Replacement and uninstall preserve durable state and project data.

Import a self-contained UTF-8 MJCF document through the authenticated Studio endpoint `POST /api/graphs/assets/mujoco` with the XML as the request body. Use the returned content reference as `mujoco_asset.asset` in the authored graph. Studio retains the exact asset in its durable web content store and resolves the same reference when opening the project, training and resuming.

## Qualify the signed MJX CPU candidate

Generate the fixture inputs with the source development environment. Copy the harness and JSON to a private QA directory outside the developer checkout. Every installed harness command uses the signed app's bundled interpreter and the exact installed fixture root.

```sh
uv run --no-sync python -m scripts.qualify_biomechanics_desktop author > "$QA/fixtures.json"
cp scripts/qualify_biomechanics_desktop.py "$QA/"
CONTENTS="$APP/Contents"
PYTHON="$CONTENTS/Resources/python/bin/python3.12"
LAUNCHER="$CONTENTS/MacOS/feedbax-studio"
```

Install the retained signed archive using its receipt and verify native background-service readiness. For the source-denial phase, stop the registered service and run the exact installed launcher with `service serve` under the host's fixture sandbox. Deny reads from the developer checkout, Homebrew and dependency stores, and deny writes inside the app. Keep the fixture state and JIT cache writable. Prove both sandbox denials before collecting results. Start a fresh service under the same sandbox after the checkpoint stop. Keep the native installation and each service-generation receipt together.

Run the harness as `"$PYTHON" -I -B "$QA/qualify_biomechanics_desktop.py"`. Pass `--contents "$CONTENTS" --fixture-root "$FIXTURE"` to commands that address the installed app. Capture each command's JSON output.

1. Run `start "$QA/fixtures.json" planar-rigid --batches 2 --checkpoint-interval 1`. Wait with `snapshot "$JOB" --status completed`, then run `checkpoint "$QA/fixtures.json" planar-rigid "$JOB"`. The checkpoint command authenticates the model slot, checks that the controller weights moved and executes the loaded model to verify its planar observation and muscle state.
2. Open the installed GUI and start `spatial-mixed` with a batch target large enough to span Quit and relaunch. Use a short checkpoint interval. Capture `snapshot "$JOB" --status running --gui present` after the first progress event, then quit the GUI and capture an absent-GUI snapshot. Reopen and capture a present-GUI snapshot. Use `--after-sequence` with the prior event sequence for each subsequent sample. Run `prove-quit` on the three snapshots to compare the service, worker, job and progress.
3. Request `"$LAUNCHER" service stop checkpoint`. With the service stopped, run `checkpoint "$QA/fixtures.json" spatial-mixed "$JOB" --source-snapshot "$QA/reopened.json"`. Restart the service and use `resume "$JOB" "$PROJECT" "$QA/fixtures.json" spatial-mixed`. Wait for the successor to progress, then let it complete or checkpoint-stop it. If stopped, use its own pre-stop snapshot for offline checkpoint inspection. Run `prove-resume` on the original and successor checkpoint records. The proof checks restart generation, checkpoint lineage, unchanged authored model, controller weight progress, XYZ observations, a normalized quaternion, and the mixed muscle-state widths.

Run `cache-probe` with the bundled interpreter under the same sandbox. It compiles a small CPU executable and verifies an actual write to the fixture's configured JIT cache. Retain the exact release and dependency identities, source-denial and native readiness logs, signed size, startup time, bundled MuJoCo notices, native inventory and cache proof. Stop and unregister the fixture after acceptance.
