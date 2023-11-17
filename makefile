.PHONY: notebooks-to-py notebooks-to-py-archive regenerate-notebooks regenerate-notebooks-archive checkout

notebooks-to-py:
	# convert all notebooks to scripts (for version control)
	@jupytext --to scripts//py:percent notebooks/*.ipynb
	@jupytext --to scripts//py:percent scratch/*.ipynb

regenerate-notebooks:
	# delete all notebooks and regenerate from jupytext scripts
	# useful when switching branches
	@find notebooks -name '*.ipynb' -xtype f -exec trash {} +
	@jupytext --set-formats ipynb,scripts//py:percent --sync notebooks/scripts/*.py
	@jupytext --set-formats ipynb,scripts//py:percent --sync scratch/scripts/*.py
	
checkout:
	@make notebooks-to-py
	@git checkout $(B)