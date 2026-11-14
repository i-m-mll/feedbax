# Feedbax Studio for macOS

Feedbax Studio uses Tauri, macOS 14 or newer on Apple Silicon, and bundled CPython 3.12.9 with MJX on JAX CPU. Each Studio bundle contains Feedbax from the same source commit and version as the release. The app reads its resources from the signed bundle and requires no developer checkout, virtual environment, or dependency installer at runtime.

The native build and Studio web styles use the signed shared window shell pinned in `vendor/window-shell`. Initialize that submodule before installing dependencies or building. For a local-only source, set the submodule URL for the checkout to the registered shell repository and initialize with Git's file transport allowed for that command. The native receipt records the pin.

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

Studio's Quit command closes the GUI. Its per-user background service continues running jobs. The service shows the active run count in the menu bar while runs are active. Choose Activity to see those runs. Stop All requests checkpoints and Option reveals Force Stop All. Quit Service requests checkpoints and shuts down the service when its jobs settle. Open Studio again to start the service.

The application menu has Reconnect, Login Items and Service Logs. A run that finishes or fails can notify you when its project window is in the background. The owner-only control socket is separate from the loopback Studio origin, and the app exchanges a short-lived ticket for the browser session.

Installed files live inside the app bundle. Durable application state lives under `~/.feedbax`, with configuration in `config`, preferences in `settings` and the default workspace under `workspace`. Project-selected output locations retain models, checkpoints and results. `~/.cache/feedbax` contains disposable compilation files, temporary files and bounded service logs. Replacement and uninstall preserve durable state and project data.
