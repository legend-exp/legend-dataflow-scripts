SHELL := /bin/bash
SOURCEDIR = source
BUILDDIR = build

all: apidoc
	sphinx-build \
      -T \
      -b html \
      -d "$(BUILDDIR)"/doctrees \
      -D language=en \
       -W --keep-going \
      "$(SOURCEDIR)" "$(BUILDDIR)"

apidoc: clean-apidoc
	sphinx-apidoc \
      --private \
      --module-first \
      --force \
      --output-dir "$(SOURCEDIR)/api" \
      ../src/legenddataflowscripts ../src/legenddataflowscripts/_version.py

clean-apidoc:
	rm -rf "$(SOURCEDIR)/api"

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"
