python_tests(
    name="tests",
    sources=["**/test_*.py"],
    overrides={
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
python_sources(name="support", sources=["**/*.py", "!**/test_*.py", "!fixtures/**/*.py"])
