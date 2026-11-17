resources(
    name="retirement_references",
    sources=["fixtures/retirement/*.json", "fixtures/retirement/*.npz"],
)
resources(name="request_corpus", sources=["fixtures/request_identity_corpus.json"])
resources(name="runpod_errors", sources=["fixtures/runpod_provider_errors/*.json"])

python_tests(
    name="tests",
    sources=["**/test_*.py"],
    overrides={
        "test_document_task_parity.py": {"dependencies": [":retirement_references"]},
        "test_document_train_pair.py": {"dependencies": [":retirement_references"]},
        "test_request_identity.py": {"dependencies": [":request_corpus"]},
        "test_runpod_transport.py": {"dependencies": [":runpod_errors"]},
        "test_dataset_figure.py": {
            "dependencies": ["web/e2e/studio/fixtures:velocity_summary_report"],
        },
        "test_project_analyze_plan.py": {
            "dependencies": ["web/e2e/studio/fixtures:velocity_analysis"],
        },
        "test_project_report_plan.py": {
            "dependencies": [
                "web/e2e/studio/fixtures:velocity_summary_report",
                "web/e2e/studio/fixtures:velocity_report_step",
            ],
        },
    },
)
python_sources(name="support", sources=["**/*.py", "!**/test_*.py"])
