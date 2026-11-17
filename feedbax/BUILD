resources(name="request_identity", sources=["contracts/request_identity_contract.json"])
resources(name="shipped_projects", sources=["examples/**/*.json", "examples/**/*.feedbax"])
python_sources(name="declarations", sources=["builtins/**/*.py"])

python_sources(
    sources=["**/*.py", "!builtins/**/*.py"],
    overrides={
        "contracts/request_identity.py": {"dependencies": [":request_identity"]},
        "examples/__init__.py": {"dependencies": [":shipped_projects"]},
        "host/client.py": {"dependencies": ["feedbax/host/__main__.py:../feedbax"]},
        "host/launcher.py": {"dependencies": ["feedbax/host/worker.py:../feedbax"]},
        "plugins/bootstrap.py": {"dependencies": [":declarations"]},
        "jobs.py": {
            "dependencies": [
                "feedbax/training_host_worker.py:feedbax",
                "feedbax/training/evaluate_worker.py:../feedbax",
                "feedbax/training/managed_studio.py:../feedbax",
                "feedbax/training/host_prewarm.py:../feedbax",
                "feedbax/plugins/composition.py:../feedbax",
            ],
        },
    },
)
