"""Publish and recover exact Loaded Models at the saved-project boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from feedbax._atomic import write_bytes_atomic
from feedbax._secure_fs import (
    DirectoryRecord, SecurePathRigor, close_directory_chain, open_directory_chain, open_existing_file,
    open_existing_path, recheck_directory_chain, validate_opened_path,
)
from feedbax.contracts.artifact_custody import ContentObjectRef, ContentRetentionTarget, ContentRootRecord
from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.loaded_model import (
    ImportedModelSource, ModelFitReport, decode_imported_model_document, encode_imported_model_document,
)
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.project import Document, LoadedModelDocument, ProjectModelArchive
from feedbax.contracts.project_resolution import LoadedModelRead, ProjectDiagnostic, ResolvedImportedModel, compose_document
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID, StoredModelDocument
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.project_subject import resolve_imported_model
from feedbax.persistence.store_damage import KnownDamage, inspect_without_recording, read_damage

if TYPE_CHECKING:
    from feedbax.component_registry import ComponentRegistry
    from feedbax.host.model_archive import UnreadableModelArchive


_ARCHIVE = TypeAdapter(ProjectModelArchive)


class ModelAssetError(ValueError):
    """The project assets cannot supply their declared Loaded Model."""


def composed_loaded_model_projection(
    loaded: LoadedModelDocument, read_document: Callable[[str], Document | None], components: ComponentRegistry,
) -> bytes | None:
    """Encode the currently composed navigation Model as a candidate for exact verification."""
    nav = loaded.model
    if nav.unresolved or nav.pointer is not None:
        return None
    document = compose_document(read_document, nav.document)
    if isinstance(document, ProjectDiagnostic) or document.schema_id != STORED_MODEL_DOCUMENT_SCHEMA_ID:
        return None
    from feedbax.compiler.model_document import complete_model_fields

    try:
        model = StoredModelDocument.model_validate(document.payload).lower(components).with_keyed_root_ports()
        return encode_imported_model_document(complete_model_fields(model, components))
    except ValueError:
        return None  # The pin may still be readable from the Store or its projection file.


def _projection_path(source: ImportedModelSource) -> str:
    return f"models/{source.model.sha256}.model.json"


@contextmanager
def _models_directory(
    root: Path, *, create: bool = False, identity: tuple[int, int] | None = None,
) -> Iterator[tuple[int, list[DirectoryRecord]]]:
    context = "Project models directory"
    chain = open_directory_chain(root, create=False, error_factory=ModelAssetError, context=context)
    try:
        parent, root_fd, info = chain[-1]
        if identity is not None and (info.st_dev, info.st_ino) != identity:
            raise ModelAssetError("The selected project folder was replaced.")
        if create:
            try:
                os.mkdir("models", dir_fd=root_fd)
            except FileExistsError:
                pass
            else:
                os.fsync(root_fd)
        fd, info = open_existing_path(
            "models", rigor=SecurePathRigor.DIRECTORY_IDENTITY, dir_fd=root_fd,
            error_factory=ModelAssetError, context=context,
        )
        chain.append((parent / "models", fd, info))
        yield fd, chain
    finally:
        close_directory_chain(chain)


def _read_asset(directory: int, path: str) -> bytes:
    name = Path(path).name
    rigor = SecurePathRigor.SINGLE_LINK_FILE_IDENTITY
    if name not in os.listdir(directory):
        raise FileNotFoundError(path)
    descriptor, before = open_existing_file(
        name, dir_fd=directory, rigor=rigor, error_factory=ModelAssetError, context=path,
    )
    with os.fdopen(descriptor, "rb") as stream:
        raw = stream.read()
        after = validate_opened_path(
            name, descriptor, dir_fd=directory, rigor=rigor, error_factory=ModelAssetError,
            context=path, expected_identity=(before.st_dev, before.st_ino),
        )
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ModelAssetError(f"{path} changed during the read.")
    return raw


def _require_pinned_model(raw: bytes, pin: ContentObjectRef, *, path: str) -> ModelDocument:
    if len(raw) != pin.size_bytes or sha256_bytes(raw) != pin.sha256:
        raise ModelAssetError(f"{path} differs from the Loaded Model's pinned Model revision.")
    return decode_imported_model_document(raw)


def write_model_assets(
    root: Path, identity: tuple[int, int], archive: str, raw: bytes, source: ImportedModelSource, projection: bytes,
    *, check: Callable[[], None],
) -> None:
    """Write both exact project files without replacing existing names."""
    assets = {_ARCHIVE.validate_python(archive): raw, _projection_path(source): projection}
    with _models_directory(root, create=True, identity=identity) as (directory, chain):
        missing = []
        for path, data in assets.items():
            try:
                present = _read_asset(directory, path)
            except FileNotFoundError:
                missing.append((path, data))
            else:
                if present != data:
                    raise FileExistsError(path)

        def guard() -> None:
            check()
            recheck_directory_chain(chain, error_factory=ModelAssetError, context="Project models directory")

        guard()
        for path, data in missing:
            try:
                write_bytes_atomic(
                    root / path, data, directory_fd=directory, replace=False, before_publish=guard,
                )
            except FileExistsError as error:
                # Another publisher may have created the file during staging.
                try:
                    same = _read_asset(directory, path) == data
                except FileNotFoundError:
                    same = False
                if not same:
                    raise FileExistsError(path) from error
        guard()


@contextmanager
def stage_project_archive(
    raw: bytes, model: ModelDocument, components: ComponentRegistry,
) -> Iterator[tuple[ImmutableArtifactBlobProvider, ImportedModelSource | ModelFitReport | UnreadableModelArchive]]:
    """Fit an archive in private custody before the caller authorizes publication."""
    from feedbax.host.model_archive import stage_model_archive

    with TemporaryDirectory(prefix="feedbax-project-model-") as directory:
        staging = ImmutableArtifactBlobProvider(Path(directory).resolve())
        with stage_model_archive(raw, model, staging, components=components) as result:
            yield staging, result


def publish_project_model(
    store_path: Path, staging: ImmutableArtifactBlobProvider, source: ImportedModelSource,
) -> None:
    """Install previously validated objects and retain their exact closure."""
    store = ImmutableArtifactBlobProvider(store_path)
    with store._exclusive_lock():
        descriptor = staging.get_tree(source.weights)
        refs = {ref.sha256: ref for ref in (
            source.model, source.weights.descriptor, *(member.object for member in descriptor.files),
        )}
        install = []
        with inspect_without_recording():
            for ref in refs.values():
                try:
                    store.verify_object(ref)
                except FileNotFoundError:
                    install.append(ref)
                else:
                    if read_damage(store, ref.sha256) is not None:
                        install.append(ref)
        for ref in install:
            store.store_object(staging.get_object(ref), expected=ref)
        store.register_root(ContentRootRecord(
            root_id=f"loaded-model-import:{source.model.sha256}:{source.weights.descriptor.sha256}",
            root_kind="import",
            targets=(
                ContentRetentionTarget(ref_kind="object", object=source.model),
                ContentRetentionTarget(ref_kind="tree", tree=source.weights),
            ),
        ))


def _read_available[T](store: ImmutableArtifactBlobProvider, read: Callable[[], T]) -> T | None:
    try:
        return read()
    except FileNotFoundError:
        return None
    except KnownDamage as error:
        # A prior Plan may have recorded the same missing object.
        if error.observation.occurrence is not None:
            raise
        try:
            with store.open_owned_object(error.observation.expected, allow_damaged=True):
                pass
        except FileNotFoundError:
            return None
        raise


def read_pinned_project_model(
    root: Path, store_path: Path, source: ImportedModelSource, navigation_model: Callable[[], bytes | None],
) -> ModelDocument:
    """Read the existing Load's pinned Model before fitting a repair upload."""
    store = ImmutableArtifactBlobProvider(store_path)
    if (raw := _read_available(store, lambda: store.get_object(source.model))) is not None:
        return decode_imported_model_document(raw)
    projection = _projection_path(source)
    try:
        with _models_directory(root) as (directory, chain):
            raw = _read_asset(directory, projection)
            recheck_directory_chain(chain, error_factory=ModelAssetError, context="Project models directory")
    except FileNotFoundError as error:
        # An explicitly repeated Load can replace a missing project projection
        # using its exact authored source.
        raw = navigation_model()
        if raw is None:
            raise ModelAssetError("The existing Load needs its pinned Model projection.") from error
        projection = "The navigation Model"
    return _require_pinned_model(raw, source.model, path=projection)


def portable_model_reader(
    root: Path, store_path: Path, components: ComponentRegistry,
) -> Callable[[LoadedModelRead], ResolvedImportedModel]:
    """Read a compiled Loaded document, recovering only absent Store bytes."""
    store = ImmutableArtifactBlobProvider(store_path)

    def read(request: LoadedModelRead) -> ResolvedImportedModel:
        source = request.source
        if (result := _read_available(store, lambda: resolve_imported_model(store, source))) is not None:
            return result
        archive = request.archive
        projection = _projection_path(source)
        try:
            with _models_directory(root) as (directory, chain):
                raw = _read_asset(directory, archive)
                authored = _read_asset(directory, projection)
                recheck_directory_chain(chain, error_factory=ModelAssetError, context="Project models directory")
        except FileNotFoundError as error:
            raise ModelAssetError(f"Loaded Model {request.document!r} needs its project assets {archive} and {projection}.") from error

        from feedbax.host.model_archive import UnreadableModelArchive

        model = _require_pinned_model(authored, source.model, path=projection)
        with stage_project_archive(raw, model, components) as (staging, candidate):
            if isinstance(candidate, UnreadableModelArchive):
                raise ModelAssetError(f"{archive}{candidate.path}: {candidate.detail}")
            if isinstance(candidate, ModelFitReport):
                raise ModelAssetError(f"{archive} does not fit the Loaded Model's pinned Model revision.")
            if candidate != source:
                raise ModelAssetError(f"{archive} differs from the Loaded Model's pinned weights or encoding.")
            resolved = resolve_imported_model(staging, source)
            publish_project_model(store_path, staging, source)
        return resolved

    return read
