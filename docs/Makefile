# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Did you install requirements/docs.txt?)
endif

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  clean      to fully remove the previous docs build"

.PHONY: clean
clean:
	rm -rf html && rm -rf src/sdp/config-docs

.PHONY: html
html:
	python gen_docs.py && $(SPHINXBUILD) src html $(SPHINXOPTS)
	@echo
	@echo "Build finished. The HTML pages are in 'html' subfolder."
