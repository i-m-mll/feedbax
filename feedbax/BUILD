resources(name="velocity_summary_figure", sources=["analysis/velocity-summary.figure.json"])
resources(name="shipped_projects", sources=["examples/**/*.json"])

python_sources(
    sources=["**/*.py"],
    overrides={
        "analysis/figure_presets.py": {"dependencies": [":velocity_summary_figure"]},
        "examples/__init__.py": {"dependencies": [":shipped_projects"]},
        "host/client.py": {"dependencies": ["feedbax/host/__main__.py:../feedbax"]},
        "host/launcher.py": {"dependencies": ["feedbax/host/worker.py:../feedbax"]},
        "jobs.py": {"dependencies": ["feedbax/training_host_worker.py:feedbax"]},
    },
)
