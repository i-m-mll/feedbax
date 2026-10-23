from __future__ import annotations

import os
from pathlib import Path

import pytest

import feedbax._secure_fs as secure_fs
import feedbax.analysis.evaluation_inputs as evaluation_inputs
import feedbax.analysis.execution_context as execution_context
import feedbax.analysis.manifest_inputs as manifest_inputs
import feedbax.orchestration.collection_recovery as collection_recovery
import feedbax.persistence.artifact_custody as artifact_custody
import feedbax.training.checkpoint_custody as checkpoint_custody


class _CallerSecurityError(ValueError):
    pass


def test_secure_path_callers_share_one_authority() -> None:
    assert artifact_custody._open_directory_chain.func is secure_fs.open_directory_chain
    assert artifact_custody._recheck_directory_chain.func is secure_fs.recheck_directory_chain
    assert execution_context.open_directory_chain is secure_fs.open_directory_chain
    assert collection_recovery.open_directory_chain is secure_fs.open_directory_chain
    assert checkpoint_custody.open_directory_chain is secure_fs.open_directory_chain
    assert manifest_inputs.open_directory_chain is secure_fs.open_directory_chain
    assert evaluation_inputs.open_directory_chain is secure_fs.open_directory_chain


def test_directory_chain_refuses_a_symlink_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    with pytest.raises(_CallerSecurityError):
        secure_fs.open_directory_chain(
            alias / "nested",
            create=True,
            error_factory=_CallerSecurityError,
            context="test directory",
        )
    assert not (real / "nested").exists()


def test_file_rigor_makes_hard_link_policy_explicit(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"payload")
    alias = tmp_path / "alias"
    os.link(original, alias)
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        descriptor, _ = secure_fs.open_existing_file(
            "original",
            rigor=secure_fs.SecurePathRigor.REGULAR_FILE_IDENTITY,
            error_factory=_CallerSecurityError,
            context="checkpoint blob",
            dir_fd=root_fd,
        )
        os.close(descriptor)
        with pytest.raises(_CallerSecurityError, match="hard-link"):
            secure_fs.open_existing_file(
                "original",
                rigor=secure_fs.SecurePathRigor.SINGLE_LINK_FILE_IDENTITY,
                error_factory=_CallerSecurityError,
                context="immutable artifact",
                dir_fd=root_fd,
            )
    finally:
        os.close(root_fd)


def test_capability_loss_fails_closed_with_the_caller_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(secure_fs.os, "O_NOFOLLOW", 0)
    with pytest.raises(_CallerSecurityError, match="O_NOFOLLOW"):
        secure_fs.require_secure_path_capabilities(
            secure_fs.SecurePathRigor.DIRECTORY_IDENTITY,
            error_factory=_CallerSecurityError,
        )


def test_cleanup_attempts_every_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[int] = []

    def close(descriptor: int) -> None:
        closed.append(descriptor)
        if descriptor == 2:
            raise OSError("close failed")

    monkeypatch.setattr(secure_fs.os, "close", close)
    with pytest.raises(OSError, match="close failed"):
        secure_fs.close_descriptors((3, 2, 1))
    assert closed == [3, 2, 1]
