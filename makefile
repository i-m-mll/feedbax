.PHONY: nb2py py2nb checkout

nb2py:
	$(MAKE) -C dev nb2py

py2nb:
	$(MAKE) -C dev py2nb
	
examples2md:
	@jupytext --to ../dev/examples_md//md examples/*.ipynb