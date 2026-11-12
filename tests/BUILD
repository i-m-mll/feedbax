python_tests(
    name="tests",
    sources=["**/test_*.py"],
    overrides={
        "test_dataset_figure.py": {
            "dependencies": ["web/e2e/studio/fixtures:velocity_summary_figure"],
        },
        "test_project_analyze_plan.py": {
            "dependencies": ["web/e2e/studio/fixtures:velocity_analysis"],
        },
    },
)
python_sources(name="support", sources=["**/*.py", "!**/test_*.py", "!fixtures/**/*.py"])
