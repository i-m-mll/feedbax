python_sources(
    sources=["**/*.py"],
    overrides={
        "host/client.py": {"dependencies": ["feedbax/host/__main__.py:../feedbax"]},
        "host/launcher.py": {"dependencies": ["feedbax/host/worker.py:../feedbax"]},
        "jobs.py": {"dependencies": ["feedbax/training_host_worker.py:feedbax"]},
    },
)
