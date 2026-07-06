.PHONY: nb2py py2nb checkout test test-all lint lint-all lint-fatal typecheck typecheck-all format ci

nb2py:
	$(MAKE) -C dev nb2py

py2nb:
	$(MAKE) -C dev py2nb

examples2md:
	@jupytext --to ../examples/markdown//md examples/*.ipynb

test:
	uv run pytest tests/test_batch_reshape_nan_bypass.py -q

test-all:
	uv run pytest tests/ -q

lint:
	uv run ruff check tests/test_batch_reshape_nan_bypass.py

lint-all:
	uv run ruff check feedbax tests

lint-fatal:
	uv run ruff check feedbax tests scripts --select E9,F63,F7,F82 --ignore F722,F821

typecheck:
	uv run python -m pyright tests/test_batch_reshape_nan_bypass.py

typecheck-all:
	uv run python -m pyright feedbax tests

format:
	uv run ruff format feedbax tests

ci:
	uv lock --check
	scripts/full_suite.sh --no-memo
	$(MAKE) lint-fatal typecheck
