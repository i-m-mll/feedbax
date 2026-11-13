"""Start the disposable Studio test backend with managed Evaluate available."""

from feedbax.web.app import create_app


def __getattr__(name: str):
    if name != "app":
        raise AttributeError(name)
    return create_app(enable_evaluation_candidate=True)


if __name__ == "__main__":
    import os
    from scripts.studio_dev import main

    if "FEEDBAX_STUDIO_TEST_OWNER_PID" not in os.environ:
        raise SystemExit("The Evaluate fixture requires an owning test runner.")
    raise SystemExit(main(backend_app="web.e2e.studio.fixtures.evaluate_series_backend:app"))
