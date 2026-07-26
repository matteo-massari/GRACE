# Sphinx documentation Makefile

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build
PORT          ?= 8000
PACKAGE_DIR   = ../src/graph_recc_sys
APIDOC_OUT    = modules

.PHONY: help clean html rebuild serve apidoc open check live full

help:
	@echo "Available targets:"
	@echo "  make html      - Build HTML documentation"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make rebuild   - Clean and rebuild HTML documentation"
	@echo "  make serve     - Serve HTML documentation locally on port $(PORT)"
	@echo "  make apidoc    - Regenerate API .rst files from source code"
	@echo "  make open      - Print path to generated index.html"
	@echo "  make check     - Build docs with warnings treated strictly"
	@echo "  make live      - Start live-reload documentation server on port $(PORT)"
	@echo "  make full      - Run apidoc + rebuild"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

rebuild: clean html

serve:
	python -m http.server $(PORT) --directory "$(BUILDDIR)/html"

apidoc:
	sphinx-apidoc -o "$(APIDOC_OUT)" "$(PACKAGE_DIR)"

open:
	@echo "Open: $(BUILDDIR)/html/index.html"

check:
	@$(SPHINXBUILD) -b html -n -W "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

live:
	@$(SPHINXAUTOBUILD) \
		--port $(PORT) \
		"$(SOURCEDIR)" "$(BUILDDIR)/html"

full: apidoc rebuild