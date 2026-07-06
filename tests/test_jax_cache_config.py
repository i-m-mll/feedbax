import subprocess


def _loaded_conftest(pytestconfig):
    for plugin in pytestconfig.pluginmanager.get_plugins():
        if str(getattr(plugin, "__file__", "")).endswith("tests/conftest.py"):
            return plugin
    raise AssertionError("tests/conftest.py plugin was not loaded")


def test_repo_cache_root_is_namespaced_by_source_fingerprint(
    monkeypatch, pytestconfig, tmp_path
) -> None:
    test_conftest = _loaded_conftest(pytestconfig)
    common_dir = tmp_path / "common"
    repo_root = tmp_path / "repo"
    calls: list[tuple[str, ...]] = []
    monkeypatch.setenv("FEEDBAX_JAX_CACHE_INVOCATION_ID", "test-invocation")

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        if args == ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"]:
            return subprocess.CompletedProcess(args, 0, stdout=f"{common_dir}\n", stderr="")
        if args == ["git", "rev-parse", "--verify", "HEAD"]:
            return subprocess.CompletedProcess(args, 0, stdout=b"abcdef1234567890\n", stderr=b"")
        if args[:5] == ["git", "diff", "--no-ext-diff", "--binary", "HEAD"]:
            return subprocess.CompletedProcess(
                args, 0, stdout=b"diff --git a/feedbax/x b/feedbax/x", stderr=b""
            )
        raise AssertionError(f"unexpected command: {args}")

    monkeypatch.setattr(test_conftest.subprocess, "run", fake_run)

    cache_root = test_conftest._repo_cache_root(repo_root)

    assert cache_root.parent.parent == common_dir / "feedbax_test_cache"
    assert cache_root.parent.name.startswith("abcdef123456-")
    assert len(cache_root.parent.name) == len("abcdef123456-") + 16
    assert cache_root.name == "test-invocation"
    assert any(call[:5] == ("git", "diff", "--no-ext-diff", "--binary", "HEAD") for call in calls)
