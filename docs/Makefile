.PHONY: html en zh_CN linkcheck clean

SPHINXBUILD ?= sphinx-build
SPHINXOPTS ?= -W --keep-going

html: en zh_CN

en:
	$(SPHINXBUILD) $(SPHINXOPTS) -c . -D language=en en _build/html/en

zh_CN:
	$(SPHINXBUILD) $(SPHINXOPTS) -c . -D language=zh_CN zh_CN _build/html/zh_CN

linkcheck:
	$(SPHINXBUILD) --keep-going -c . -b linkcheck en _build/linkcheck/en

clean:
	$(RM) -r _build
