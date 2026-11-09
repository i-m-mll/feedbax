# Feedbax Studio for macOS

The base release uses Tauri, macOS 14 or newer on Apple Silicon, and bundled CPython 3.12.9 with JAX on CPU. The app reads its resources from the signed bundle. It requires no developer checkout, virtual environment, or dependency installer at runtime.

## Build a private candidate

From a clean commit on an Apple Silicon build Mac with Xcode command-line tools, Rust, Bun and uv, install the two locked Bun environments and run the native and bundle phases.

```sh
(cd desktop && bun install --frozen-lockfile)
(cd web && bun install --frozen-lockfile)
uv run --no-project --with packaging==26.3 python desktop/package.py native
uv run --no-project --with packaging==26.3 python desktop/package.py bundle \
  --output .desktop-build/candidate --build-number 1 \
  --signing-identity "$STUDIO_SIGNING_IDENTITY"
```

Use an exact Apple Development certificate name or fingerprint for a private candidate. Passing `-` requests ad-hoc signing explicitly. Both are private build classes. Public distribution requires Developer ID, notarization and quarantine first-launch qualification. Allow a separate bounded window for the first Cargo compile.

The native phase uses Cargo's normal target reuse. Bundle assembly checks the recorded native inputs and launcher digest, then runs Tauri's bundle phase without recompilation. The release directory contains the signed ZIP and a receipt binding it to the source commit, native payload, wheel closure and signing identity. Use a new output path and increment the build number for a replacement release.

For a private fixture, add `--fixture-name <unique-name> --fixture-root <absolute-path>` to the bundle phase. Both releases in a replacement exercise must use the same fixture name and state root. Keep the absolute root short enough for the macOS Unix socket address limit.

The numerical package join supplies `--numerical-manifest <path>`. The manifest uses `feedbax.desktop.wheels.v1`, targets `aarch64-apple-darwin`, declares `runtime` and lists exact wheel filenames and SHA-256 values under `wheels`. Stage the wheel files alongside the manifest. These replace same-name locked wheels or add new wheels. The final package still has to pass dependency, native-code and installed runtime acceptance.

For the stock biomechanics CPU build, add `--runtime-profile mjx-cpu` and supply the qualified numerical manifest. The recipe adds the locked `mjx` extra. The manifest must attest the exact macOS wheels for `jax`, `jaxlib`, `mujoco-core` and `mujoco-mjx`. This profile accepts only wheels from the selected lock closure and reports the MJX JAX provider on CPU. The default `jax-cpu` profile retains the base dependency set.

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
