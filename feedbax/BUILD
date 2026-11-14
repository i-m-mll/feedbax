resources(name="velocity_summary_figure", sources=["analysis/velocity-summary.figure.json"])
resources(name="request_identity", sources=["contracts/request_identity_contract.json"])
resources(name="shipped_projects", sources=["examples/**/*.json", "examples/**/*.feedbax"])

python_sources(
    sources=["**/*.py"],
    overrides={
        "analysis/figure_presets.py": {"dependencies": [":velocity_summary_figure"]},
        "contracts/request_identity.py": {"dependencies": [":request_identity"]},
        "examples/__init__.py": {"dependencies": [":shipped_projects"]},
        "host/client.py": {"dependencies": ["feedbax/host/__main__.py:../feedbax"]},
        "host/launcher.py": {"dependencies": ["feedbax/host/worker.py:../feedbax"]},
        "jobs.py": {
            "dependencies": [
                "feedbax/training_host_worker.py:feedbax",
                "feedbax/training/evaluate_worker.py:../feedbax",
                "feedbax/training/managed_studio.py:../feedbax",
                "feedbax/plugins/composition.py:../feedbax",
            ],
        },
    },
)
